#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Βοηθητική συνάρτηση για αρχικοποίηση με μη-μηδενικούς ακεραίους
void init_poly(int *poly, int size) {
    for (int i = 0; i < size; i++) {
        int val = (rand() % 10) + 1; // Τυχαίοι από 1 έως 10
        if (rand() % 2 == 0) val = -val; // Τυχαίο πρόσημο
        poly[i] = val;
    }
}

int main(int argc, char *argv[]) {
    int rank, size;
    int n; // Βαθμός πολυωνύμου
    
    // Μεταβλητές χρονομέτρησης
    double t_send_start = 0.0, t_send_end = 0.0, t_calc_start = 0.0, t_calc_end = 0.0;
    double t_recv_start = 0.0, t_recv_end = 0.0, t_total_start = 0.0, t_total_end = 0.0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 2) {
        if (rank == 0) printf("Usage: %s <degree n>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    n = atoi(argv[1]);
    int terms = n + 1; // Πλήθος όρων (π.χ. βαθμός 2 -> ax^2+bx+c -> 3 όροι)

    // Δηλώσεις δεικτών
    int *A = NULL, *B = NULL;       // Τα αρχικά πολυώνυμα (μόνο στο rank 0 αρχικά)
    long long *global_C = NULL;     // Το τελικό αποτέλεσμα (μόνο στο rank 0)
    int *local_A = NULL;            // Το κομμάτι του A που θα πάρει κάθε διεργασία
    long long *local_C = NULL;      // Το μερικό αποτέλεσμα κάθε διεργασίας

    // --- Βήμα 0: Αρχικοποίηση (Δεν μετράει στον χρόνο) ---
    
    // Το B το θέλουν όλοι ολόκληρο, δεσμεύουμε μνήμη παντού
    B = (int *)malloc(terms * sizeof(int));
    
    // Υπολογισμός κομματιών για το scatterv
    int *sendcounts = NULL;
    int *displs = NULL;

    if (rank == 0) {
        srand(time(NULL));
        A = (int *)malloc(terms * sizeof(int));
        init_poly(A, terms);
        init_poly(B, terms);
        
        // Το αποτέλεσμα θα έχει βαθμό 2n, άρα 2n+1 όρους
        global_C = (long long *)calloc((2 * n + 1), sizeof(long long));

        // Υπολογισμός κατανομής φορτίου (Scatterv parameters)
        sendcounts = (int *)malloc(size * sizeof(int));
        displs = (int *)malloc(size * sizeof(int));

        int remainder = terms % size;
        int sum = 0;
        for (int i = 0; i < size; i++) {
            sendcounts[i] = terms / size;
            if (i < remainder) sendcounts[i]++;
            displs[i] = sum;
            sum += sendcounts[i];
        }
    }

    // Διαμοιρασμός του μεγέθους του chunk που θα πάρει ο καθένας
    int my_A_size;
    
    // Πρέπει να ξέρουμε το μέγεθος local_A. Το υπολογίζουμε τοπικά:
    int remainder = terms % size;
    my_A_size = terms / size + (rank < remainder ? 1 : 0);

    local_A = (int *)malloc(my_A_size * sizeof(int));
    
    // Το local C θα έχει μέγεθος: (μέγεθος_local_A - 1) + (μέγεθος_B - 1) + 1 = my_A_size + n
    int my_C_size = my_A_size + n;
    local_C = (long long *)calloc(my_C_size, sizeof(long long));

    // Περιμένουμε όλες τις διεργασίες για να αρχίσει η χρονομέτρηση
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) t_total_start = MPI_Wtime(); // Έναρξη συνολικού χρόνου

    // --- Βήμα 1: Αποστολή Δεδομένων (Send Time) ---
    if (rank == 0) t_send_start = MPI_Wtime();

    // 1. Broadcast το B σε όλους
    MPI_Bcast(B, terms, MPI_INT, 0, MPI_COMM_WORLD);

    // 2. Scatter το A σε κομμάτια
    MPI_Scatterv(A, sendcounts, displs, MPI_INT, local_A, my_A_size, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) t_send_end = MPI_Wtime();

    // --- Βήμα 2: Παράλληλος Υπολογισμός (Calc Time) ---
    MPI_Barrier(MPI_COMM_WORLD); // Για δίκαιη μέτρηση του υπολογισμού
    if (rank == 0) t_calc_start = MPI_Wtime();

    // Naive Polynomial Multiplication: Local A * Global B
    for (int i = 0; i < my_A_size; i++) {
        for (int j = 0; j < terms; j++) {
            // Ο όρος i του local_A αντιστοιχεί σε δύναμη x^(displs[rank] + i) αν το βλέπαμε καθολικά.
            // local_A[i] * B[j] συνεισφέρει στον όρο i+j του local_C
            local_C[i + j] += (long long)local_A[i] * B[j];
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) t_calc_end = MPI_Wtime();

    // --- Βήμα 3: Λήψη Αποτελεσμάτων (Receive Time) ---
    if (rank == 0) t_recv_start = MPI_Wtime();

    if (rank == 0) {
        // 1. Πρόσθεσε τα αποτελέσματα του Rank 0 στο global_C
        for (int i = 0; i < my_C_size; i++) {
            global_C[displs[0] + i] += local_C[i];
        }

        // Βελτίωση: Δέσμευση μνήμης ΜΙΑ φορά έξω από το loop
        int max_recv_size = (terms / size + 1) + n; 
        long long *temp_buf = (long long *)malloc(max_recv_size * sizeof(long long));

        // 2. Λήψη από τους υπόλοιπους
        for (int p = 1; p < size; p++) {
            int remote_A_size = sendcounts[p];
            int remote_C_size = remote_A_size + n;
            
            // Λήψη στο temp_buf
            MPI_Recv(temp_buf, remote_C_size, MPI_LONG_LONG, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // Πρόσθεση στο τελικό διάνυσμα
            // Η λογική είναι: Το κομμάτι Α της διεργασίας p ξεκινούσε στη θέση displs[p].
            int offset = displs[p];
            for (int k = 0; k < remote_C_size; k++) {
                global_C[offset + k] += temp_buf[k];
            }
        }
        free(temp_buf); // Αποδέσμευση στο τέλος
    } else {
        // Οι workers στέλνουν το αποτέλεσμά τους
        MPI_Send(local_C, my_C_size, MPI_LONG_LONG, 0, 0, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        t_recv_end = MPI_Wtime();
        t_total_end = MPI_Wtime(); // Τέλος συνολικού χρόνου

        // --- Εκτύπωση Χρόνων ---
        printf("Results for N=%d, Processes=%d\n", n, size);
        printf("(i) Data Distr Time: %.9f sec\n", t_send_end - t_send_start);
        printf("(ii) Calculation Time: %.9f sec\n", t_calc_end - t_calc_start);
        printf("(iii) Gather Time: %.9f sec\n", t_recv_end - t_recv_start);
        printf("(iv) Total Time: %.9f sec\n", t_total_end - t_total_start);
        
        // Προαιρετικά: Εκτύπωση αποτελεσμάτων για μικρά n για επαλήθευση
        if (n < 10) {
            printf("Result C: ");
            for (int i = 0; i <= 2 * n; i++) printf("%lld ", global_C[i]);
            printf("\n");
        }
        
        free(A); 
        free(global_C); 
        free(sendcounts); 
        free(displs);
    }

    free(B); 
    free(local_A); 
    free(local_C);

    MPI_Finalize();
    return 0;
}
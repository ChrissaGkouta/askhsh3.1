import subprocess
import re


degrees = [20000, 40000, 80000, 100000] 
processes = [1, 2, 4, 8]
runs_per_exp = 4

def parse_output(output):
    # Εξαγωγή των χρόνων 
    times = {}
    for line in output.split('\n'):
        if "Send Time" in line: times['send'] = float(line.split(':')[1].strip().split()[0])
        if "Calc Time" in line: times['calc'] = float(line.split(':')[1].strip().split()[0])
        if "Recv Time" in line: times['recv'] = float(line.split(':')[1].strip().split()[0])
        if "Total Time" in line: times['total'] = float(line.split(':')[1].strip().split()[0])
    return times

# Συνάρτηση για υπολογισμό μέσου όρου χωρίς numpy
def calculate_mean(data_list):
    if not data_list:
        return 0.0
    return sum(data_list) / len(data_list)

print(f"{'N':<10} {'Procs':<10} {'Avg Total (s)':<15} {'Avg Calc (s)':<15} {'Avg Comm (s)':<15}")
print("-" * 70)

for n in degrees:
    for p in processes:
        results = {'send': [], 'calc': [], 'recv': [], 'total': []}
        
        for _ in range(runs_per_exp):
            cmd = ["mpirun", "--oversubscribe", "-np", str(p), "./poly_mult", str(n)]
            try:
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    print(f"Error executing N={n}, P={p}")
                    continue
                    
                times = parse_output(result.stdout))
                
                if times:
                    for k, v in times.items():
                        results[k].append(v)
            except Exception as e:
                print(f"Exception running MPI: {e}")

        # Υπολογισμός μέσων όρων 
        if results['total']:
            avg_total = calculate_mean(results['total'])
            avg_calc = calculate_mean(results['calc'])
            avg_comm = calculate_mean(results['send']) + calculate_mean(results['recv'])
            
            print(f"{n:<10} {p:<10} {avg_total:<15.4f} {avg_calc:<15.4f} {avg_comm:<15.4f}")
        else:
            print(f"{n:<10} {p:<10} {'FAILED':<15} {'FAILED':<15} {'FAILED':<15}")

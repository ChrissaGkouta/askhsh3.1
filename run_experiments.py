import subprocess
import re
import numpy as np

# Παράμετροι πειραμάτων
degrees = [1000, 5000, 10000] # Ξεκίνα με μικρά. Το 10^5 είναι βαρύ για O(n^2)
processes = [1, 2, 4, 8]
runs_per_exp = 4

def parse_output(output):
    # Εξαγωγή των χρόνων από το stdout
    times = {}
    for line in output.split('\n'):
        if "Send Time" in line: times['send'] = float(line.split(':')[1].strip().split()[0])
        if "Calc Time" in line: times['calc'] = float(line.split(':')[1].strip().split()[0])
        if "Recv Time" in line: times['recv'] = float(line.split(':')[1].strip().split()[0])
        if "Total Time" in line: times['total'] = float(line.split(':')[1].strip().split()[0])
    return times

print(f"{'N':<10} {'Procs':<10} {'Avg Total (s)':<15} {'Avg Calc (s)':<15} {'Avg Comm (s)':<15}")
print("-" * 70)

for n in degrees:
    for p in processes:
        results = {'send': [], 'calc': [], 'recv': [], 'total': []}
        
        for _ in range(runs_per_exp):
            # Εκτέλεση της εντολής MPI
            cmd = ["mpirun", "-np", str(p), "./poly_mult", str(n)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"Error executing N={n}, P={p}")
                continue
                
            times = parse_output(result.stdout)
            for k, v in times.items():
                results[k].append(v)
        
        # Υπολογισμός μέσων όρων
        avg_total = np.mean(results['total'])
        avg_calc = np.mean(results['calc'])
        avg_comm = np.mean(results['send']) + np.mean(results['recv'])
        
        print(f"{n:<10} {p:<10} {avg_total:<15.4f} {avg_calc:<15.4f} {avg_comm:<15.4f}")
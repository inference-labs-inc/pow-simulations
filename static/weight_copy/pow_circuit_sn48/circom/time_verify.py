import time
import subprocess

start_time = time.time()

verify_cmd = "snarkjs groth16 verify verification_key.json public.json proof.json"
subprocess.run(verify_cmd, shell=True)

end_time = time.time()
elapsed_time = end_time - start_time

print(f"\nVerification time: {elapsed_time:.4f} seconds")

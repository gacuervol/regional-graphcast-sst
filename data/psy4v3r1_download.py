# Get the current user name
import os, sys
# Obtener el nombre del usuario actual
if sys.platform.startswith("linux"):
    usuario_actual = os.path.expanduser("~").split('/')[-1]
elif sys.platform.startswith("win"):
    usuario_actual = os.path.expanduser("~").split('\\')[-1]
else:
    raise EnvironmentError("Unsupported OS")

import os
from ftplib import FTP

# Connect to the FTP server
ftp = FTP('ftp.mercator-ocean.fr') 
ftp.login(user=, passwd=)

# List all files in the directory
all_files_listed = ftp.nlst()
print(f":: Total files: {len(all_files_listed)}")
print(f":: First 5 files: {all_files_listed[:5]}")

# remove '.' and '..' from the list
all_files_listed = [file for file in all_files_listed if file not in ['.', '..']]
print(f":: Total files: {len(all_files_listed)}")
print(f":: First 5 files: {all_files_listed[:5]}")

# Define the local directory where the files will be downloaded
local_dir = f'/home/{usuario_actual}/repo/data/psy4v3r1_data/'
for file in all_files_listed:
    print(f":: Downloading {file}...")
    local_path = os.path.join(local_dir, file)
    with open(local_path, 'wb') as f:
        ftp.retrbinary(f'RETR {file}', f.write)
    print(f":: Downloaded {file} to {local_path}")

print(":: All files downloaded successfully!")

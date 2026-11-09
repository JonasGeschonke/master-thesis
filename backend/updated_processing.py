import os, subprocess, sys, requests

# Set the directory path
directory = './db/pre'

# Set the output directory path
out_unvoiced_dir = './db/post/unvoiced'
out_voiced_dir = './db/post/voiced'

# url = 'https://ssi-gatherer.pb.ax:14080'
url = 'http://localhost:14080'

# Ensure the directory exists
os.makedirs(out_unvoiced_dir, exist_ok=True)
os.makedirs(out_voiced_dir, exist_ok=True)

processes = {}

uuids: list[str] = requests.get(url + '/audio/uuids').json()
if (len(uuids) == 0):
    print('No recordings to be converted')
    exit(1)

print(f'########\n Recordings to be converted: {uuids} \n')

# Reverse the list of uuids to process the most recent ones first
uuids.reverse()

# Look for duplicate uuids in the list and add them to a separate list
duplicate_uuids = []
seen_uuids = set()

for uuid in uuids:
    if uuid in seen_uuids:
        duplicate_uuids.append(uuid)
        # Remove the duplicate uuid from the list of uuids to be processed
        uuids.remove(uuid)
    else:
        seen_uuids.add(uuid)

print(f'########\n Duplicate uuids: {duplicate_uuids} \n')

for uuid in uuids:
    print(f'Current uuid: {uuid}')
    path_unvoiced = directory + '/' + 'unvoiced_' + uuid + '.webm'
    path_voiced = directory + '/' + 'voiced_' + uuid + '.webm'

    path_flac_unvoiced = out_unvoiced_dir + '/' + 'unvoiced_' + uuid + '.flac'
    path_flac_voiced = out_voiced_dir + '/' + 'voiced_' + uuid + '.flac'

    processes[uuid] = {
        'unvoiced_done': False,
        'voiced_done': False,
        'path_unvoiced': path_flac_unvoiced,
        'path_voiced': path_flac_voiced
    }

    if os.path.exists(path_flac_unvoiced) and os.path.exists(path_flac_voiced):
        print(f'Flac files for uuid {uuid} already exist, skipping conversion')
        processes[uuid]['unvoiced_done'] = True
        processes[uuid]['voiced_done'] = True
    elif os.path.exists(path_unvoiced) and os.path.exists(path_voiced):
        # Run unvoiced command
        try:
            return_code = subprocess.run(
                ['ffmpeg', '-i', path_unvoiced, '-c:a', 'flac', path_flac_unvoiced],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT
            )

            if return_code.returncode != 0:
                processes[uuid]['unvoiced_done'] = False
            else:
                processes[uuid]['unvoiced_done'] = True

        except subprocess.CalledProcessError as e:
            print(e.output.decode('utf-8'))

        # Run voiced command
        try:
            return_code = subprocess.run(
                ['ffmpeg', '-i', path_voiced, '-c:a', 'flac', path_flac_voiced],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT
            )

            if return_code.returncode != 0:
                processes[uuid]['voiced_done'] = False
            else:
                processes[uuid]['voiced_done'] = True

        except subprocess.CalledProcessError as e:
            print(e.output.decode('utf-8'))
    else:
        print(f'Files for uuid {uuid} do not exist')

print('########\n Updating Database: \n')
for uuid in processes:
    if processes[uuid]['unvoiced_done'] and processes[uuid]['voiced_done']:
        form = {
            'unvoiced_flac_path': processes[uuid]['path_unvoiced'],
            'voiced_flac_path': processes[uuid]['path_voiced']
        } 
        requests.post(url + f'/recordings/{uuid}', data=form)
        print(f'Updated uuid: {uuid}')
    else:
        print(f'Failed to update uuid: {uuid}')
# Starting Docker
```bash
docker run -u 0 -it -v ./db/:/data mmcauliffe/montreal-forced-aligner:latest
```
# Download model
```bash
mfa model download acoustic english_mfa
mfa model download dictionary english_mfa
```
# Run the conversion script
```bash
python mfa_pre_processing.py
```
# Do the Alignment
```bash
mfa align --clean --single_speaker /data/alignments english_mfa english_mfa /data/alignments/out
```
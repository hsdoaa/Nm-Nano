#note user should provide the coordinate file and the fastq file
import subprocess
import sys
import argparse

#https://stackoverflow.com/questions/21579892/using-argparse-to-for-filename-input 
# Initialize parser
parser = argparse.ArgumentParser("start of getting RNA fastq files")

#reference file argument
parser.add_argument('-r', action='store', dest='ref_input', help='Provide reference genome file')

#fastq file argument 
parser.add_argument('-f', action='store', dest='fastq_input', help='Provide Fastq file')
 

# Read arguments from command line
args = parser.parse_args()

sys.argv[2] = args.ref_input
sys.argv[4] = args.fastq_input
##################################################################################

with open('hek.sam','w') as f:
    subprocess.run(["minimap2", "-ax", "map-ont", "--split-prefix", "/tmp/temp_name", sys.argv[2], sys.argv[4]], stdout=f) 
subprocess.run(["nanopolish" ,"index", "-d", "fast5_files/", sys.argv[4]])
with open('hek.bam','w') as f1:
    subprocess.run(["samtools" ,"view", "-S", "-b", "hek.sam"],stdout=f1)
subprocess.run(["samtools" ,"sort", "hek.bam", "-o", "hek.sorted.bam"])
subprocess.run(["samtools" ,"index", "hek.sorted.bam"])
subprocess.run(["samtools","quickcheck" ,"hek.sorted.bam"])
with open('hek-reads-ref.eventalign.txt','w') as f3:
    subprocess.run(["nanopolish","eventalign" ,"--reads", sys.argv[4], "--bam", "hek.sorted.bam","--genome", sys.argv[2], "--scale-events"], stdout=f3)
subprocess.run(["python","gen_coors_Nm.py"])
subprocess.run(["python","extract_nm.py",])

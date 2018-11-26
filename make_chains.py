"""Script to generate the chains for the likelihoods."""
import os.path
import argparse
from likelihood import LikelihoodClass

parser = argparse.ArgumentParser()
parser.add_argument('--quadratic', action='store_true', help='Use the quadratic emulator class')
parser.add_argument('--noemuerr', action='store_true', help='Exclude emulator error')
parser.add_argument('testdir', type=str, help='Directory to use for testing')
args = parser.parse_args()

testdata=os.path.join(args.testdir, "output")
#Get a savefile
root, testdir = os.path.split(args.testdir)
if testdir == '':
    testdir = os.path.split(root)
savefile = "chain_"+testdir +".txt"
if args.noemuerr:
    savefile += "-noemuerr"
#Build the emulator
if args.quadratic:
    savefile = os.path.join("simulations/hires_s8_quadratic", savefile)
    like = LikelihoodClass(basedir=os.path.expanduser("simulations/hires_s8_quadratic"), emulator_class="quadratic")
else:
    savefile = os.path.join("simulations/hires_s8", savefile)
    like = LikelihoodClass(basedir=os.path.join("simulations/hires_s8"))

print("Save to", savefile)
output = like.do_sampling(savefile, datadir=testdata, include_emulator_error = not args.noemuerr)

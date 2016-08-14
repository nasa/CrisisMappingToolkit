# -----------------------------------------------------------------------------
# Copyright * 2014, United States Government, as represented by the
# Administrator of the National Aeronautics and Space Administration. All
# rights reserved.
#
# The Crisis Mapping Toolkit (CMT) v1 platform is licensed under the Apache
# License, Version 2.0 (the "License"); you may not use this file except in
# compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# total_lake_run.py Code Summary 
#
# This code doesn't take any inputs and instead it automatically
# breaks up the lakes into 3 different, even groups of lakes. 
# The code then starts threads and runs all of the lakes denoted
# by the input into Lake_Level_Run
#
# MAJOR ISSUES WITH total_lake_run.py!!!!
# When you press ctrl+C it goes to cancel the threads but for some reason
# the four threads that are being allowed to run by semaphore (all in lake_measure.py 
# in cmt/lake_helper) are cancelled, but then the new 1,000 lakes go run to take 
# over so you can't successfully close out of total_lake_run.py. This means 
# The first 4 threads will close successfully, but until you kill the python code
# the other threads will try to run. Killing them will cause them to just   
# clean out the entire file. :(
#
# -----------------------------------------------------------------------------


import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
from cmt.lake_helper.lake_measure import * 
import signal
import threading



# This function and function call to cancel threads naturally if ctrl+C is used
def signal_handler(signal, frame):
    print('You pressed Ctrl+C!')
    Lake_Level_Cancel()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)



def main():
	# Total lakes goes up to 3721, excludes lake id 4001, Fallen Lake 
	numlakes = 3721

	lake1_start = 150 # where we start the first group, right now skipping 1st 150 because they're too big
	lake2_start = numlakes//3 # Where we start the second group
	lake3_start = (numlakes//3)*2 # Where we start the third group

	# Creating the list, split up into 3 sections 
	lakeids_1 = range(lake1_start,lake2_start)
	lakeids_2 = range(lake2_start,lake3_start)
	lakeids_3 = range(lake3_start,numlakes+1)
	lakeids_3.append(4001) # Adding on lake 4001, for some reason 3772-4000 are skipped

	# Convert ints to strings
	lakeids_1 = map(str, lakeids_1)
	lakeids_2 = map(str, lakeids_2)
	lakeids_3 = map(str, lakeids_3)

	# Start & end dates
	date = '1983-10-01'
	enddate = '2017-12-01'

	Lake_Level_Run(lakeids_3, date, enddate,'results',fai=False, ndti=False, \
		update_function = None, complete_function = None)



t = threading.Thread(target=main)
t.start()

while t.is_alive():
	t.join(1)



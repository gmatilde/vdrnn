import os
import pickle
import argparse

import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres

from hpbandster.optimizers import H2BO

from Worker import VRNN_worker as worker

import logging

logging.basicConfig(level=logging.DEBUG)

parser = argparse.ArgumentParser(description='run BOHB to optimize VRNN')
parser.add_argument('--min_budget', type=float, help='Minimum number of epochs for training.', default=60)
parser.add_argument('--max_budget', type=float, help='Maximum number of epochs for training.', default=600)
parser.add_argument('--n_iterations', type=int,   help='Number of iterations performed by the optimizer', default=1000)
parser.add_argument('--worker', help='Flag to turn this into a worker process', action='store_true')
parser.add_argument('--run_id', type=str, help='A unique run id for this optimization run. An easy option is to use the job id of the clusters scheduler.')
parser.add_argument('--nic_name', type=str, help='Which network interface to use for communication.', default='lo')
parser.add_argument('--shared_directory', type=str, help='A directory that is accessible for all processes, e.g. a NFS share.', default='.')

args=parser.parse_args()
# Every process has to lookup the hostname
host = hpns.nic_name_to_host(args.nic_name)

if args.worker:
	w = worker('mnist.json', path='/home/matilde/Documents/vdrnn/surrogate_data', run_id=args.run_id, host=host)
	w.load_nameserver_credentials(working_directory=args.shared_directory)
	w.run(background=False)
	exit(0)

result_logger = hpres.json_result_logger(directory=args.shared_directory, overwrite=True)

# Start a nameserver:
NS = hpns.NameServer(run_id=args.run_id, host=host, port=0, working_directory=args.shared_directory)
ns_host, ns_port = NS.start()

# Start local worker
w = worker('mnist.json', path='/home/matilde/Documents/vdrnn/surrogate_data', run_id=args.run_id, host=host, nameserver=ns_host, nameserver_port=ns_port)
w.run(background=True)

# Run an optimizer
bohb = H2BO(  configspace = worker.get_configspace(),
			  run_id = args.run_id,
              eta=2,
			  host=host,
			  nameserver=ns_host,
			  nameserver_port=ns_port,
			  result_logger=result_logger,
			  min_budget=args.min_budget, max_budget=args.max_budget,
		   )

res = bohb.run(n_iterations=args.n_iterations)

# shutdown
bohb.shutdown(shutdown_workers=True)
NS.shutdown()

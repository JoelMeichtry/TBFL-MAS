from Agents.ServerAgent import ServerAgent
from Agents.ClientAgent import ClientAgent
from Utilities.Argparser import Argparser
import asyncio
from Utilities.Data import Data
import logging
from Utilities.Metrics import Metrics
import os
from Utilities.Paths import config, Paths
import spade
import uuid
import yaml

fsm_logger = logging.getLogger("FSM")
agents = {}

async def main(launch_config={}):
    args = Argparser.args_parser()
    batch_size_options = config["options"]["batch_size_per_class"]

    opt = vars(args)
    opt.update(launch_config)

    batch_size_per_classes_server = {}
    Data.prepare_batch_sizes_per_classes(args, batch_size_options,
                                    batch_size_per_classes_server, 100)

    server = ServerAgent(args.jid_server, "server", args,
                         batch_size_per_classes_server, 100, 200)
    agents["server"] = server
    await server.start(auto_register=True)
    fsm_logger.info(args.jid_server + " is created")
    if args.algorithm != "ML":
        for i in range(args.number_of_client_agents):
            batch_size_per_classes = {}
            Data.prepare_batch_sizes_per_classes(args, batch_size_options, batch_size_per_classes, i * 13)
            # name_of_client = "client_" + str(uuid.uuid4())[:6]
            name_of_client = "client" + str(i)
            agent = ClientAgent(name_of_client + "@localhost", name_of_client, args,
                                batch_size_per_classes, i * 7, i * 9)
            agents[name_of_client] = agent
            await agent.start(auto_register=True)
            fsm_logger.info(name_of_client + " is created")

    first = True
    while agents["server"].is_alive():
        try:
            await asyncio.sleep(1)
            # simulate new entry of an agent
            if agents["server"].args.epoch == 2 and args.new_entry_or_leave == "new entry" and first:
                batch_size_per_classes = {}
                Data.prepare_batch_sizes_per_classes(args, batch_size_options, batch_size_per_classes, 20)
                name_of_client = "client_" + str(uuid.uuid4())[:6]
                agent = ClientAgent(name_of_client + "@localhost", name_of_client, args,
                                    batch_size_per_classes, 20, 30)
                agents[name_of_client] = agent
                await agent.start(auto_register=True)
                fsm_logger.info(name_of_client + " is created")
                first = False
            # simulate leave of an agent
            elif agents["server"].args.epoch == 2 and args.new_entry_or_leave == "leave" and first:
                agents["client0"].presence.unsubscribe(args.jid_server)
                print("client0 leaves the MAS")
                fsm_logger.info("client0 leaves the MAS")
                first = False
        except KeyboardInterrupt:
            break

async def multiple_mains():
    args = Argparser.args_parser()
    path_to_learning_scenarios_config = str(Paths.get_project_root()) + "\\Configuration\\learning_scenarios_config.yml"
    learning_scenarios_conf = yaml.load(open(path_to_learning_scenarios_config), Loader=yaml.FullLoader)
    for learning_scenarios in args.launch_config:
        for learning_scenario in learning_scenarios:
            print("This run uses the learning scenario {}".format(learning_scenario))
            await main(launch_config=learning_scenarios_conf["learning_scenarios"][learning_scenario])
            print(learning_scenario + " is done")

def plot():
    args = Argparser.args_parser()
    for key in agents.keys():
        Metrics.plot_metrics(args, key, key, "test_acc")
        Metrics.plot_metrics(args, key, key, "test_f1")
        Metrics.plot_metrics(args, key, key, "test_pre")
        Metrics.plot_metrics(args, key, key, "test_rec")
        Metrics.plot_metrics(args, key, key, "loss")
        for i in range(args.number_of_classes_in_dataset):
            Metrics.plot_metrics(args, key, key, "f1_cla" + str(i))
            Metrics.plot_metrics(args, key, key, "pre_cla" + str(i))
            Metrics.plot_metrics(args, key, key, "rec_cla" + str(i))
        Metrics.plot_metrics(args, key, key, "batch_sizes_per_classes")

if __name__ == "__main__":
    spade.run(multiple_mains())
    plot()
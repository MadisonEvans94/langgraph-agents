./uninstall.sh && sleep 10
#./dev_agent.sh -t -b -d $1
#./dev_agent.sh -t -d $1
./dev_agent.sh -t --vllm-config default_stream.yaml -d $1

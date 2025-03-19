help() {
  echo "Usage: ./dev_agent.sh [OPTION] ..."
  echo "Options are: "
  echo -e "\t -b, --build \t\t\t build and push images"
  echo -e "\t -d, --deploy-agent [NAME] \t deploy agent with optional name"
  echo -e "\t --vllm-config [CONFIG] \t specify vllm config"
  echo -e "\t -t, --test \t\t\t test the deployed agent"
  echo -e "\t -h, --help \t\t\t print help"    
}

_build=false
_deploy_agent=false
_vllm_config=""
_deploy_name=as
_uninstall=false
_test=false
DEPLOY_NS=ogpt

if [ ! -n "$1" ]; then
    help
    exit
fi

while [ -n "$1" ];
do
  case $1 in
    -b  | --build)
          _build=true
          ;;
    -u  | --uninstall)
          _uninstall=true
          ;;
    -d  | --deploy-agent)
        _deploy_agent=true
        shift
        if [ -n "$1" ]; then
          _deploy_name=$1
        fi
        ;;
    --deploy-vllm)
        _deploy_vllm=true
        ;;
    --vllm-config)
        shift
        if [ -n "$1" ]; then
          _vllm_config=$1
        fi
        ;;
    -t  | --test)
        _test=true
        ;;
     -h  | --help)
        help
        exit 0
        ;;        
    * )
        echo "The parameter $1 is not allowed"
        echo
        help
        exit 1
        ;;
    esac
    shift
done

print_flags() {
  echo "Build: $_build"
  echo "Deploy Agent: $_deploy_agent"
  echo "Deploy VLLM: $_deploy_vllm"
  echo "VLLM Config: $_vllm_config"
  echo "Uninstall: $_uninstall"
}

print_flags

if [ "$_build" = true ]; then
  echo "Building and pushing images..."
  pushd ../
  docker build -t sapdai/refd:agent-service-v3 -f Dockerfile.agent-service .
  docker push sapdai/refd:agent-service-v3
  popd
fi

if [ "$_deploy_agent" = true ]; then
  echo "Deploying agent..."
  helm upgrade --install ${_deploy_name} agent-stack --namespace $DEPLOY_NS --wait --timeout 30s
fi

if [ "$_test" = true ]; then
  app=${_deploy_name}-agent-client
  client_pod=$(kubectl get pods -l app=$app --namespace $DEPLOY_NS --no-headers | awk '{print $1}')
  target_service_host=$(kubectl get pod $client_pod -o jsonpath="{.spec.containers[0].env[?(@.name=='TARGET_SERVICE_HOST')].value}")
  target_service_port=$(kubectl get pod $client_pod -o jsonpath="{.spec.containers[0].env[?(@.name=='TARGET_SERVICE_PORT')].value}")
  url="$target_service_host:$target_service_port/ask"
  
  json_payload='{ "agent_type": "conversational_agent_with_routing", "user_query": "hi" }'
  echo "Testing agent..."
  kubectl exec $client_pod --namespace $DEPLOY_NS -- \
    curl --no-buffer -s $url -X POST -d "$json_payload" -H 'Content-Type: application/json'  
fi

if [ "$_uninstall" = true ]; then
  echo "Uninstalling agent..."
  helm uninstall ${_deploy_name} --namespace $DEPLOY_NS
fi

# todo add image name
# todo add config name

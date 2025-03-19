_deploy_name=as
DEPLOY_NS=ogpt
app=${_deploy_name}-agent-server
serv_pod=$(kubectl get pods -l app=$app --namespace $DEPLOY_NS --no-headers | awk '{print $1}')
echo $serv_pod
kubectl exec -it $serv_pod -- /bin/bash

# coral-app


# usefull commands
export KUBECONFIG="./workload01-kubeconfig (2).conf"

kubectl delete pods -n coral-project-8lxdv --all
kubectl rollout restart deployment coral-app -n coral-project-8lxdv
kubectl get pods -n coral-project-8lxdv



Last 100% commit: 386c729cd3a10993c0f250dcf8537e4b465da066

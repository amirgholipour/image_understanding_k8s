# image_understanding_k8s
This project is aimed to do various tasks related to image understanding and publish the solution as a Microservices with K8S.


Here's a step-by-step guide on deploying a Docker microservice using Kubernetes:

    1. Create a Docker image of your microservice: Ensure that your microservice is containerized using Docker. Create a Dockerfile that describes your application, its dependencies, and how it should be executed.

    2. Push the Docker image to a container registry: Push your Docker image to a container registry like Docker Hub, Google Container Registry, or Amazon ECR. This will make the image accessible to your Kubernetes cluster.

    php

docker build -t <username>/<image-name>:<tag> .
docker login
docker push <username>/<image-name>:<tag>

    3. Create a Kubernetes deployment configuration file: Create a YAML file (e.g., deployment.yaml) to describe your Kubernetes Deployment. This file should include the following information:

    The Docker image to use (the one you pushed to the container registry)
    The desired number of replicas for your microservice
    Port mapping and any environment variables needed by your application



    4. Apply the deployment configuration: Run the following command to create the Deployment in your Kubernetes cluster:

kubectl apply -f deployment.yaml

    5. Create a Kubernetes service configuration file: Create a YAML file (e.g., service.yaml) to describe your Kubernetes Service. This file should include the following information:

    The type of service (ClusterIP, NodePort, LoadBalancer, or ExternalName)
    The target port of your microservice (the port your application is listening on)
    Any necessary selector labels to associate the service with the correct pods



    6. Apply the service configuration: Run the following command to create the Service in your Kubernetes cluster:

kubectl apply -f service.yaml

    7. Access your microservice: Once the deployment and service are running, you can access your microservice using the Kubernetes node's IP address and the NodePort (if you chose the NodePort type). You can find the NodePort using kubectl get svc.

arduino

    kubectl get svc
    curl http://<node-ip>:<node-port>/your-endpoint

That's it! Your Docker microservice should now be deployed and accessible via Kubernetes. Remember to replace <username>/<image-name>:<tag>, <node-ip>, and <node-port> with the appropriate values for your setup.


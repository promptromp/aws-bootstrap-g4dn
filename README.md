# aws-bootstrap-g4dn

This repository contains code and documentation to make it fast and easy to bootstrap an AWS EC2 instance running a Deep Learning AMI (e.g. Ubuntu or Amazon Linux) with a CUDA-compliant Nvidia GPU (e.g. g4dn.xlarge by default).

The idea is to make it easy in particular to quickly spawn cost-effective Spot Instances via AWS CLI , bootstrapping the instance with an SSH key, and ramping up to be able to develop using CUDA.

Main workflows we're optimizing for are hybrid local-remote workflows e.g.:

1. Using Jupyter server-client (with Jupyter server running on the instance and local jupyter client)
2. Using VSCode Remote SSH extension
3. Using Nvidia Nsight for remote debugging


## Additional Resources

For pricing information on GPU instances see [here](https://instances.vantage.sh/aws/ec2/g4dn.xlarge).
Deep Learning AMIs - see [here](https://docs.aws.amazon.com/dlami/latest/devguide/what-is-dlami.html)
Nvidia Nsight - Setup Remote Debugging - see [here](https://docs.nvidia.com/nsight-visual-studio-edition/3.2/Content/Setup_Remote_Debugging.htm)

A couple of additional relevant recent tutorials (2025) for setting up CUDA environment on EC2 GPU instances are:

https://www.dolthub.com/blog/2025-03-12-provision-an-ec2-gpu-host-on-aws/
https://techfortalk.co.uk/2025/10/11/aws-ec2-setup-for-gpu-cuda-programming/

# Setup
Set up your system as detailed [here](https://github.com/krai/ck-qaic/blob/main/script/setup.docker/README.md).

# Benchmarking
```
SDK_VER=v1.6.80 POWER=no SUT=edge_r282_z93_q8 DOCKER=yes MULTISTREAM_ONLY=yes WORKLOADS="ssd_resnet34" $(ck find ck-qaic:script:run)/run_edge.sh
```
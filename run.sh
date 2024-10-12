#!/bin/bash
echo "Download MNIST"
python hudes/websocket_server.py --run-in thread --download-dataset-and-exit
echo "Done downloading MNIST"
python hudes/websocket_server.py --device mps --run-in thread &
server_pid=$!
python hudes/hudes_play.py --input keyboardGL
python hudes/hudes_play.py --input keyboard
#python hudes/hudes_play.py --input xtouch 
kill ${server_pid}
kill -9 ${server_pid}

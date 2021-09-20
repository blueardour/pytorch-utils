
rm output.onnx

export PYTHONPATH="$HOME/workspace/git/onnx-simplifier-master":$PYTHONPATH
python -V

python custom_layers.py

python -c 'import onnx; print("onnx version: ", onnx.__version__); import onnxsim; print("onnxsim version: ", onnxsim.__version__)'
python -m onnxsim 'output.onnx' 'output-update.onnx'

#python test.py -o 'output-update.onnx'

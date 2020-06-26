python mnist_eva.py --defense km --model LeNet --attack-type pgd --eps 0.1 
python mnist_eva.py --defense km --model LeNet --attack-type pgd --eps 0.15
python mnist_eva.py --defense km --model LeNet --attack-type pgd --eps 0.2
python mnist_eva.py --defense km --model LeNet --attack-type pgd --eps 0.25
python mnist_eva.py --defense km --model LeNet --attack-type pgd --eps 0.3
python mnist_eva.py --defense km --model LeNet --attack-type pgd --eps 0.35

python mnist_eva.py --defense km --model A --attack-type pgd --eps 0.1 
python mnist_eva.py --defense km --model A --attack-type pgd --eps 0.15
python mnist_eva.py --defense km --model A --attack-type pgd --eps 0.2
python mnist_eva.py --defense km --model A --attack-type pgd --eps 0.25
python mnist_eva.py --defense km --model A --attack-type pgd --eps 0.3
python mnist_eva.py --defense km --model A --attack-type pgd --eps 0.35

python mnist_eva.py --defense km --model B --attack-type pgd --eps 0.1 
python mnist_eva.py --defense km --model B --attack-type pgd --eps 0.15
python mnist_eva.py --defense km --model B --attack-type pgd --eps 0.2
python mnist_eva.py --defense km --model B --attack-type pgd --eps 0.25
python mnist_eva.py --defense km --model B --attack-type pgd --eps 0.3
python mnist_eva.py --defense km --model B --attack-type pgd --eps 0.35

python mnist_eva.py --defense km --model C --attack-type pgd --eps 0.1 
python mnist_eva.py --defense km --model C --attack-type pgd --eps 0.15
python mnist_eva.py --defense km --model C --attack-type pgd --eps 0.2
python mnist_eva.py --defense km --model C --attack-type pgd --eps 0.25
python mnist_eva.py --defense km --model C --attack-type pgd --eps 0.3
python mnist_eva.py --defense km --model C --attack-type pgd --eps 0.35

# python mnist_eva.py --defense km --model LeNet --attack-type pgd --iter 50 --eps 0.3 --alpha 0.01 --fname pgd
# python mnist_eva.py --defense km --model A --attack-type pgd --iter 50 --eps 0.3 --alpha 0.01 --fname pgd
# python mnist_eva.py --defense km --model B --attack-type pgd --iter 50 --eps 0.3 --alpha 0.01 --fname pgd
# python mnist_eva.py --defense km --model C --attack-type pgd --iter 50 --eps 0.3 --alpha 0.01 --fname pgd

# python mnist_eva.py --defense bs --model LeNet --attack-type pgd --iter 50 --eps 0.3 --alpha 0.01 --fname pgd
# python mnist_eva.py --defense bs --model A --attack-type pgd --iter 50 --eps 0.3 --alpha 0.01 --fname pgd
# python mnist_eva.py --defense bs --model B --attack-type pgd --iter 50 --eps 0.3 --alpha 0.01 --fname pgd
# python mnist_eva.py --defense bs --model C --attack-type pgd --iter 50 --eps 0.3 --alpha 0.01 --fname pgd

# python mnist_eva.py --defense ms --model LeNet --attack-type pgd --iter 50 --eps 0.3 --alpha 0.01 --fname pgd
# python mnist_eva.py --defense ms --model A --attack-type pgd --iter 50 --eps 0.3 --alpha 0.01 --fname pgd
# python mnist_eva.py --defense ms --model B --attack-type pgd --iter 50 --eps 0.3 --alpha 0.01 --fname pgd
# python mnist_eva.py --defense ms --model C --attack-type pgd --iter 50 --eps 0.3 --alpha 0.01 --fname pgd

# python mnist_eva.py --defense jf --model LeNet --attack-type pgd --iter 50 --eps 0.3 --alpha 0.01 --fname pgd
# python mnist_eva.py --defense jf --model A --attack-type pgd --iter 50 --eps 0.3 --alpha 0.01 --fname pgd
# python mnist_eva.py --defense jf --model B --attack-type pgd --iter 50 --eps 0.3 --alpha 0.01 --fname pgd
# python mnist_eva.py --defense jf --model C --attack-type pgd --iter 50 --eps 0.3 --alpha 0.01 --fname pgd

# python mnist_eva.py --defense km --model LeNet --attack-type deepfool --iter 100 --fname deepfool
# python mnist_eva.py --defense km --model A --attack-type deepfool --iter 100 --fname deepfool
# python mnist_eva.py --defense km --model B --attack-type deepfool --iter 100 --fname deepfool
# python mnist_eva.py --defense km --model C --attack-type deepfool --iter 100 --fname deepfool

# python mnist_eva.py --defense bs --model LeNet --attack-type deepfool --iter 100 --fname deepfool
# python mnist_eva.py --defense bs --model A --attack-type deepfool --iter 100 --fname deepfool
# python mnist_eva.py --defense bs --model B --attack-type deepfool --iter 100 --fname deepfool
# python mnist_eva.py --defense bs --model C --attack-type deepfool --iter 100 --fname deepfool

# python mnist_eva.py --defense ms --model LeNet --attack-type deepfool --iter 100 --fname deepfool
# python mnist_eva.py --defense ms --model A --attack-type deepfool --iter 100 --fname deepfool
# python mnist_eva.py --defense ms --model B --attack-type deepfool --iter 100 --fname deepfool
# python mnist_eva.py --defense ms --model C --attack-type deepfool --iter 100 --fname deepfool

# python mnist_eva.py --defense jf --model LeNet --attack-type deepfool --iter 100 --fname deepfool
# python mnist_eva.py --defense jf --model A --attack-type deepfool --iter 100 --fname deepfool
# python mnist_eva.py --defense jf --model B --attack-type deepfool --iter 100 --fname deepfool
# python mnist_eva.py --defense jf --model C --attack-type deepfool --iter 100 --fname deepfool

# python mnist_eva.py --defense km --model LeNet --attack-type deepfool --iter 100 --fname deepfool
# python mnist_eva.py --defense km --model A --attack-type deepfool --iter 100 --fname deepfool
# python mnist_eva.py --defense km --model B --attack-type deepfool --iter 100 --fname deepfool
# python mnist_eva.py --defense km --model C --attack-type deepfool --iter 100 --fname deepfool

# python mnist_eva.py --defense bs --model LeNet --attack-type deepfool --iter 100 --fname deepfool
# python mnist_eva.py --defense bs --model A --attack-type deepfool --iter 100 --fname deepfool
# python mnist_eva.py --defense bs --model B --attack-type deepfool --iter 100 --fname deepfool
# python mnist_eva.py --defense bs --model C --attack-type deepfool --iter 100 --fname deepfool

# python mnist_eva.py --defense ms --model LeNet --attack-type deepfool --iter 100 --fname deepfool
# python mnist_eva.py --defense ms --model A --attack-type deepfool --iter 100 --fname deepfool
# python mnist_eva.py --defense ms --model B --attack-type deepfool --iter 100 --fname deepfool
# python mnist_eva.py --defense ms --model C --attack-type deepfool --iter 100 --fname deepfool

# python mnist_eva.py --defense jf --model LeNet --attack-type deepfool --iter 100 --fname deepfool
# python mnist_eva.py --defense jf --model A --attack-type deepfool --iter 100 --fname deepfool
# python mnist_eva.py --defense jf --model B --attack-type deepfool --iter 100 --fname deepfool
# python mnist_eva.py --defense jf --model C --attack-type deepfool --iter 100 --fname deepfool
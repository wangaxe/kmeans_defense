# python mnist_eva.py --defense km --model LeNet --attack-type pgd --eps 0.1 
# python mnist_eva.py --defense km --model LeNet --attack-type pgd --eps 0.15
# python mnist_eva.py --defense km --model LeNet --attack-type pgd --eps 0.2
# python mnist_eva.py --defense km --model LeNet --attack-type pgd --eps 0.25
# python mnist_eva.py --defense km --model LeNet --attack-type pgd --eps 0.3
# python mnist_eva.py --defense km --model LeNet --attack-type pgd --eps 0.35
# python mnist_eva.py --defense km --model LeNet --attack-type pgd --eps 0.40
# python mnist_eva.py --defense km --model LeNet --attack-type pgd --eps 0.45

# python mnist_eva.py --defense km --model A --attack-type pgd --eps 0.1 
# python mnist_eva.py --defense km --model A --attack-type pgd --eps 0.15
# python mnist_eva.py --defense km --model A --attack-type pgd --eps 0.2
# python mnist_eva.py --defense km --model A --attack-type pgd --eps 0.25
# python mnist_eva.py --defense km --model A --attack-type pgd --eps 0.3
# python mnist_eva.py --defense km --model A --attack-type pgd --eps 0.35
# python mnist_eva.py --defense km --model A --attack-type pgd --eps 0.4
# python mnist_eva.py --defense km --model A --attack-type pgd --eps 0.45

# python mnist_eva.py --defense km --model B --attack-type pgd --eps 0.1 
# python mnist_eva.py --defense km --model B --attack-type pgd --eps 0.15
# python mnist_eva.py --defense km --model B --attack-type pgd --eps 0.2
# python mnist_eva.py --defense km --model B --attack-type pgd --eps 0.25
# python mnist_eva.py --defense km --model B --attack-type pgd --eps 0.3
# python mnist_eva.py --defense km --model B --attack-type pgd --eps 0.35
# python mnist_eva.py --defense km --model B --attack-type pgd --eps 0.4
# python mnist_eva.py --defense km --model B --attack-type pgd --eps 0.45

# python mnist_eva.py --defense km --model C --attack-type pgd --eps 0.1 
# python mnist_eva.py --defense km --model C --attack-type pgd --eps 0.15
# python mnist_eva.py --defense km --model C --attack-type pgd --eps 0.2
# python mnist_eva.py --defense km --model C --attack-type pgd --eps 0.25
# python mnist_eva.py --defense km --model C --attack-type pgd --eps 0.3
# python mnist_eva.py --defense km --model C --attack-type pgd --eps 0.35
# python mnist_eva.py --defense km --model C --attack-type pgd --eps 0.4
# python mnist_eva.py --defense km --model C --attack-type pgd --eps 0.45

# python mnist_eva.py --defense km --model LeNet --attack-type fgsm --eps 0.3 --fname fgsm
# python mnist_eva.py --defense km --model A --attack-type fgsm --eps 0.3 --fname fgsm
# python mnist_eva.py --defense km --model B --attack-type fgsm --eps 0.3 --fname fgsm
# python mnist_eva.py --defense km --model C --attack-type fgsm --eps 0.3 --fname fgsm

# python mnist_eva.py --defense bs --model LeNet --attack-type fgsm --eps 0.3 --fname fgsm
# python mnist_eva.py --defense bs --model A --attack-type fgsm --eps 0.3 --fname fgsm
# python mnist_eva.py --defense bs --model B --attack-type fgsm --eps 0.3 --fname fgsm
# python mnist_eva.py --defense bs --model C --attack-type fgsm --eps 0.3 --fname fgsm

# python mnist_eva.py --defense ms --model LeNet --attack-type fgsm --eps 0.3 --fname fgsm
# python mnist_eva.py --defense ms --model A --attack-type fgsm --eps 0.3 --fname fgsm
# python mnist_eva.py --defense ms --model B --attack-type fgsm --eps 0.3 --fname fgsm
# python mnist_eva.py --defense ms --model C --attack-type fgsm --eps 0.3 --fname fgsm 

# python mnist_eva.py --defense jf --model LeNet --attack-type fgsm --eps 0.3 --fname fgsm
# python mnist_eva.py --defense jf --model A --attack-type fgsm --eps 0.3 --fname fgsm
# python mnist_eva.py --defense jf --model B --attack-type fgsm --eps 0.3 --fname fgsm
# python mnist_eva.py --defense jf --model C --attack-type fgsm --eps 0.3 --fname fgsm

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

python mnist_white_noise.py --model LeNet --defense km --fname clean
python mnist_white_noise.py --model LeNet --defense bs --fname clean
python mnist_white_noise.py --model LeNet --defense jf --fname clean
python mnist_white_noise.py --model LeNet --defense ms --fname clean
python mnist_white_noise.py --model A --defense km --fname clean
python mnist_white_noise.py --model A --defense bs --fname clean
python mnist_white_noise.py --model A --defense jf --fname clean
python mnist_white_noise.py --model A --defense ms --fname clean
python mnist_white_noise.py --model B --defense km --fname clean
python mnist_white_noise.py --model B --defense bs --fname clean
python mnist_white_noise.py --model B --defense jf --fname clean
python mnist_white_noise.py --model B --defense ms --fname clean
python mnist_white_noise.py --model C --defense km --fname clean
python mnist_white_noise.py --model C --defense bs --fname clean
python mnist_white_noise.py --model C --defense jf --fname clean
python mnist_white_noise.py --model C --defense ms --fname clean
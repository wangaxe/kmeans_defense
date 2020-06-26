# python svhn_eva.py --defense km --attack-type fgsm --eps 8 --model pr18 --fname fgsm
# python svhn_eva.py --defense km --attack-type fgsm --eps 8 --model vgg11 --fname fgsm
# python svhn_eva.py --defense km --attack-type fgsm --eps 8 --model vgg16 --fname fgsm

# python svhn_eva.py --defense km --attack-type fgsm --eps 16 --model pr18 --fname fgsm
# python svhn_eva.py --defense km --attack-type fgsm --eps 16 --model vgg11 --fname fgsm
# python svhn_eva.py --defense km --attack-type fgsm --eps 16 --model vgg16 --fname fgsm

# python svhn_eva.py --defense bs --attack-type fgsm --eps 8 --model pr18 --fname fgsm
# python svhn_eva.py --defense bs --attack-type fgsm --eps 8 --model vgg11 --fname fgsm
# python svhn_eva.py --defense bs --attack-type fgsm --eps 8 --model vgg16 --fname fgsm

# python svhn_eva.py --defense bs --attack-type fgsm --eps 16 --model pr18 --fname fgsm
# python svhn_eva.py --defense bs --attack-type fgsm --eps 16 --model vgg11 --fname fgsm
# python svhn_eva.py --defense bs --attack-type fgsm --eps 16 --model vgg16 --fname fgsm

# python svhn_eva.py --defense ms --attack-type fgsm --eps 8 --model pr18 --fname fgsm
# python svhn_eva.py --defense ms --attack-type fgsm --eps 8 --model vgg11 --fname fgsm
# python svhn_eva.py --defense ms --attack-type fgsm --eps 8 --model vgg16 --fname fgsm

# python svhn_eva.py --defense ms --attack-type fgsm --eps 16 --model pr18 --fname fgsm
# python svhn_eva.py --defense ms --attack-type fgsm --eps 16 --model vgg11 --fname fgsm
# python svhn_eva.py --defense ms --attack-type fgsm --eps 16 --model vgg16 --fname fgsm

# python svhn_eva.py --defense jf --attack-type fgsm --eps 8 --model pr18 --fname fgsm
# python svhn_eva.py --defense jf --attack-type fgsm --eps 8 --model vgg11 --fname fgsm
# python svhn_eva.py --defense jf --attack-type fgsm --eps 8 --model vgg16 --fname fgsm

# python svhn_eva.py --defense jf --attack-type fgsm --eps 16 --model pr18 --fname fgsm
# python svhn_eva.py --defense jf --attack-type fgsm --eps 16 --model vgg11 --fname fgsm
# python svhn_eva.py --defense jf --attack-type fgsm --eps 16 --model vgg16 --fname fgsm

##############################################
python svhn_eva.py --defense km --attack-type pgd --eps 8 --model pr18 --fname pgd
python svhn_eva.py --defense km --attack-type pgd --eps 8 --model vgg11 --fname pgd
python svhn_eva.py --defense km --attack-type pgd --eps 8 --model vgg16 --fname pgd

python svhn_eva.py --defense km --attack-type pgd --eps 16 --model pr18 --fname pgd
python svhn_eva.py --defense km --attack-type pgd --eps 16 --model vgg11 --fname pgd
python svhn_eva.py --defense km --attack-type pgd --eps 16 --model vgg16 --fname pgd

python svhn_eva.py --defense bs --attack-type pgd --eps 8 --model pr18 --fname pgd
python svhn_eva.py --defense bs --attack-type pgd --eps 8 --model vgg11 --fname pgd
python svhn_eva.py --defense bs --attack-type pgd --eps 8 --model vgg16 --fname pgd

python svhn_eva.py --defense bs --attack-type pgd --eps 16 --model pr18 --fname pgd
python svhn_eva.py --defense bs --attack-type pgd --eps 16 --model vgg11 --fname pgd
python svhn_eva.py --defense bs --attack-type pgd --eps 16 --model vgg16 --fname pgd

python svhn_eva.py --defense ms --attack-type pgd --eps 8 --model pr18 --fname pgd
python svhn_eva.py --defense ms --attack-type pgd --eps 8 --model vgg11 --fname pgd
python svhn_eva.py --defense ms --attack-type pgd --eps 8 --model vgg16 --fname pgd

python svhn_eva.py --defense ms --attack-type pgd --eps 16 --model pr18 --fname pgd
python svhn_eva.py --defense ms --attack-type pgd --eps 16 --model vgg11 --fname pgd
python svhn_eva.py --defense ms --attack-type pgd --eps 16 --model vgg16 --fname pgd

python svhn_eva.py --defense jf --attack-type pgd --eps 8 --model pr18 --fname pgd
python svhn_eva.py --defense jf --attack-type pgd --eps 8 --model vgg11 --fname pgd
python svhn_eva.py --defense jf --attack-type pgd --eps 8 --model vgg16 --fname pgd

python svhn_eva.py --defense jf --attack-type pgd --eps 16 --model pr18 --fname pgd
python svhn_eva.py --defense jf --attack-type pgd --eps 16 --model vgg11 --fname pgd
python svhn_eva.py --defense jf --attack-type pgd --eps 16 --model vgg16 --fname pgd

##################################################
python svhn_eva.py --defense km --attack-type deepfool --iter 100 --model pr18 --fname deepfool
python svhn_eva.py --defense km --attack-type deepfool --iter 100 --model vgg11 --fname deepfool
python svhn_eva.py --defense km --attack-type deepfool --iter 100 --model vgg16 --fname deepfool

python svhn_eva.py --defense bs --attack-type deepfool --iter 100 --model pr18 --fname deepfool
python svhn_eva.py --defense bs --attack-type deepfool --iter 100 --model vgg11 --fname deepfool
python svhn_eva.py --defense bs --attack-type deepfool --iter 100 --model vgg16 --fname deepfool

python svhn_eva.py --defense ms --attack-type deepfool --iter 100 --model pr18 --fname deepfool
python svhn_eva.py --defense ms --attack-type deepfool --iter 100 --model vgg11 --fname deepfool
python svhn_eva.py --defense ms --attack-type deepfool --iter 100 --model vgg16 --fname deepfool

python svhn_eva.py --defense jf --attack-type deepfool --iter 100 --model pr18 --fname deepfool
python svhn_eva.py --defense jf --attack-type deepfool --iter 100 --model vgg11 --fname deepfool
python svhn_eva.py --defense jf --attack-type deepfool --iter 100 --model vgg16 --fname deepfool
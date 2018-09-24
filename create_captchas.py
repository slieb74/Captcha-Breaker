from claptcha import Claptcha
from PIL import Image
import numpy as np
import time, os, itertools


number_list = ['0','1','2','3','4','5','6','7','8','9']
alphabet_lowercase = ['a','b','c','d','e','f','g','h','i','j','k','l','m',
'n','o','p','q','r','s','t','u','v','w','x','y','z']
alphabet_uppercase = ['A','B','C','D','E','F','G','H','I','J','K','L','M',
'N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
ch_list = number_list + alphabet_lowercase + alphabet_uppercase

ch_dict={}
for index, ch in enumerate(ch_list):
    ch_dict[ch] = index

ch_dict_2 = {}
combos = itertools.product(ch_list, ch_list)
for index, combo in enumerate(combos):
    ch_dict_2[''.join(combo)]=index

with open('train_2.txt', 'w') as t:
    for path in os.listdir('captcha_output_data2/multi_char_train_data'):
        t.write('/Users/SamLiebman/flatiron/captcha/captcha_output_data2/multi_char_train_data/'+str(path) + ', ' + str(ch_dict_2.get(path.split('_')[0])) + ', ' + 'train' + '\n')
    for path in os.listdir('captcha_output_data2/multi_char_test_data'):
        t.write('/Users/SamLiebman/flatiron/captcha/captcha_output_data2/multi_char_test_data/'+str(path) + ', ' + str(ch_dict_2.get(path.split('_')[0])) + ', ' + 'test' + '\n')
    for path in os.listdir('captcha_output_data2/multi_char_val_data'):
        t.write('/Users/SamLiebman/flatiron/captcha/captcha_output_data2/multi_char_val_data/'+str(path) + ', ' + str(ch_dict_2.get(path.split('_')[0])) + ', ' + 'val' + '\n')



# Create text file with class names
if False:
    with open("class_names.txt2", "w") as output:
        for index, ch in enumerate(ch_list):
            output.write(str(ch) + '_' + str(index) +'\n')

    # with open('train.txt2', 'w') as t:
    #     for path in os.listdir('captcha_output_data2/multi_char_train_data'):
    #         t.write('/Users/SamLiebman/flatiron/captcha/captcha_output_data2/multi_char_train_data/'+str(path) + ' ' +  str(np.random.randint(0,5)) + ',' +  str(np.random.randint(0,5)) +  ',' + str(56 - np.random.randint(0,5)) +  ',' + str(28 - np.random.randint(0,5)) +
    #         ',' + str(ch_dict.get(path.split('_')[0])) + '\n')

    with open('image_paths2.txt', 'w') as f:
        for path in os.listdir('captcha_output_data2/multi_char_train_data'):
            f.write(str(path) + '\n')

    with open('anchor_paths2.txt', 'w') as a:
        for path in os.listdir('captcha_output_data2/multi_char_train_data'):
            a.write(str(np.random.randint(0,5)) + ',' +  str(np.random.randint(0,5)) +  ',' + str(28 - np.random.randint(0,5)) +  ',' + str(28 - np.random.randint(0,5)) +
            ',' + str(ch_dict.get(path.split('_')[0]))+'\n')

if False:
    def generate_single_char_captchas(ch_list, num_samples=1000, size=(28,28),
                                      margin=(0,0), noise=np.random.random()):
        test_labels=[]
        train_labels=[]
        val_labels=[]

        for ch in ch_list:
            for n in range(num_samples):
                c = Claptcha(ch, "OpenSans-Regular.ttf", noise=noise, size=size, margin=margin)

                #test data
                if n % 5 == 0:
                    if ch.isupper():
                        text, file = c.write('captcha_output_data/single_char_test_data/{}_upper_{}.png'.format(ch,n))
                    elif ch.islower():
                        text, file = c.write('captcha_output_data/single_char_test_data/{}_lower_{}.png'.format(ch,n))
                    else:
                        text, file = c.write('captcha_output_data/single_char_test_data/{}_{}.png'.format(ch,n))

                    test_labels.append((text, file))

                #train/val data
                else:
                    if n % 8 == 0:
                        if ch.isupper():
                            text, file = c.write('captcha_output_data/single_char_val_data/{}_upper_{}.png'.format(ch,n))
                        elif ch.islower():
                            text, file = c.write('captcha_output_data/single_char_val_data/{}_lower_{}.png'.format(ch,n))
                        else:
                            text, file = c.write('captcha_output_data/single_char_val_data/{}_{}.png'.format(ch,n))

                        val_labels.append((text,file))

                    else:
                        if ch.isupper():
                            text, file = c.write('captcha_output_data/single_char_train_data/{}_upper_{}.png'.format(ch,n))
                        elif ch.islower():
                            text, file = c.write('captcha_output_data/single_char_train_data/{}_lower_{}.png'.format(ch,n))
                        else:
                            text, file = c.write('captcha_output_data/single_char_train_data/{}_{}.png'.format(ch,n))

                        train_labels.append((text, file))

        return test_labels, train_labels, val_labels

if True:
    def generate_multi_char_captchas(ch_list, num_chars=[2], num_samples=100, margin=(5,5), noise=np.random.random()/2):

        test_labels=[]
        train_labels=[]
        val_labels=[]

        for num in num_chars:
            for i in ch_list:
                for j in ch_list:
                    for n in range(num_samples):
                        chars = i+j
                        c = Claptcha(chars, "OpenSans-Regular.ttf", noise=noise, size=(28*num,28), margin=margin)

                        #test data
                        if n % 5 == 0:
                            text, file = c.write('captcha_output_data2/multi_char_test_data/{}_{}.png'.format(chars,n))
                            test_labels.append((text, file))

                        #train/val data
                        else:
                            if n % 8 == 0:
                                text, file = c.write('captcha_output_data2/multi_char_val_data/{}_{}.png'.format(chars,n))
                                val_labels.append((text,file))

                            else:
                                text, file = c.write('captcha_output_data2/multi_char_train_data/{}_{}.png'.format(chars,n))
                                train_labels.append((text, file))

                        if (n+1) % 5000 == 0:
                            print('Runtime: {} seconds. {}/{} samples generated'.format(round(time.time()-start,2), n+1, num_samples*len(num_chars)))
        return test_labels, train_labels, val_labels

# test_labels, train_labels, val_labels = generate_multi_char_captchas(ch_list, num_chars=[2], num_samples=20, margin=(5,5), noise=np.random.random()/2)

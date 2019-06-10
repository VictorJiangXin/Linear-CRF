import codecs  
import sys  
  
def character_tagging(input_file, output_file):  
    input_data = codecs.open(input_file, 'r', 'utf-8')  
    output_data = codecs.open(output_file, 'w', 'utf-8')  
    for line in input_data.readlines():  
        word_list = line.strip().split()  
        for word in word_list:  
            if len(word) == 1:  
                output_data.write(word + "\tS\n")  
            else:  
                output_data.write(word[0] + "\tB\n")  
                for w in word[1:len(word)-1]:  
                    output_data.write(w + "\tI\n")  
                output_data.write(word[len(word)-1] + "\tE\n")  
        output_data.write("\n")  
    input_data.close()  
    output_data.close()
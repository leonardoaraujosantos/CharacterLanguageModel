import torch
import torch.distributions
import numpy as np
import utils_char_dataset
import model as models

def getNextChar(chars, num_chars, model, device, codemap, greedly=True):        
    chars_class = [utils_char_dataset.class_id_from_char(char, codemap) for char in chars]
    len_input = len(chars_class)    
    input = torch.tensor(chars_class).type(torch.LongTensor).unsqueeze(0).to(device)    
    len_input = torch.tensor(len_input).unsqueeze(0).to(device)    
    curr_batch_size = len_input.shape[0]            
    model.eval()
    pred_class_lst = []
    with torch.no_grad():
        hidden_state = models.initHidden(curr_batch_size, False, model.hidden_size, model.num_layers, device)
        # If more than one character were passed
        if len_input > 1:            
            probabilities, hidden_state = model(input, hidden_state, len_input)
            if greedly:
                _, pred_class = torch.max(probabilities, dim=2)            
            else:
                # Sample from the probabilities distribution
                m = torch.distributions.Categorical(probabilities)
                pred_class = m.sample()
            
            input = pred_class[0][-1].unsqueeze(0).unsqueeze(0)
            pred_class_lst.append(input.item())
        # Return greedly the next char
        for idx in range(num_chars):            
            probabilities, hidden_state = model(input, hidden_state, torch.tensor(1).unsqueeze(0))     
            if greedly:
                _, pred_class = torch.max(probabilities, dim=2)            
            else:
                m = torch.distributions.Categorical(probabilities)
                pred_class = m.sample()          
            
            input = pred_class
            pred_class_lst.append(input.item())
            if pred_class.item() == utils_char_dataset.class_id_from_char(utils_char_dataset.EOS_token, codemap):
                break                            
    
    return pred_class_lst


def getProbabilitySentence(word, model, device, codemap):        
    # Convert each character on the word into it's class id
    chars_class = [utils_char_dataset.class_id_from_char(char, codemap) for char in word]
    num_chars = len(chars_class)        
    curr_batch_size = 1            
    model.eval()
    scores_lst = []
    with torch.no_grad():
        # Initialize model on the beginning of the sequence
        hidden_state = models.initHidden(curr_batch_size, False, model.hidden_size, model.num_layers, device)
        # Return greedly the next char
        for idx in range(num_chars):
            # Convert class word index to a tensor
            input = torch.tensor(chars_class[idx]).type(torch.LongTensor).unsqueeze(0).unsqueeze(0).to(device)   
            # Push input(character) to the model
            probabilities, hidden_state = model(input, hidden_state, torch.tensor(1).unsqueeze(0))
            # Get the probability of the next input character given the previous inputs 
            if idx < num_chars - 1:
                input_next = torch.tensor(chars_class[idx+1]).type(torch.LongTensor).unsqueeze(0).unsqueeze(0).to(device).item()                
                probabilities = probabilities.squeeze(0).squeeze(0)    
                prob_next = probabilities[input_next].item()
                scores_lst.append(prob_next)                   
            
    # Return the product of the probabilities
    return np.prod(scores_lst)
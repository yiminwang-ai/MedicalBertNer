from torch import nn
from transformers import BertForTokenClassification


class BertModel(nn.Module):
    def __init__(self,num_labels):
        super(BertModel,self).__init__()
        self.bert=BertForTokenClassification.from_pretrained('BertModel',num_labels=num_labels)
    def forward(self,input_ids,attention_mask,token_type_ids,labels):
        output=self.bert(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids,labels=labels,
                         output_hidden_states=False,output_attentions=False,return_dict=False)
        return output




import matplotlib.pyplot as plt
import torch
import torchvision
from IPython.display import clear_output

from torch import nn
from torchvision import transforms

#============================================PATCH EMBEDDING=========================================
class PatchEmbedding(nn.Module):
    def __init__(self, 
                 in_channels:int=3,
                 patch_size:int=16,
                 embedding_dim:int=768):
        super().__init__()
        
        self.patch_size=patch_size
        self.patcher = nn.Conv2d(in_channels=in_channels,
                                 out_channels=embedding_dim,
                                 kernel_size=patch_size,
                                 stride=patch_size,
                                 padding=0)

        self.flatten = nn.Flatten(start_dim=2,end_dim=3)

    def forward(self, x):
        image_resolution = x.shape[-1]
        assert image_resolution % self.patch_size == 0, f"Input image size must be divisble by patch size, image shape: {image_resolution}, patch size: {patch_size}"
        
        x_patched = self.patcher(x)
        x_flattened = self.flatten(x_patched) 
        return x_flattened.permute(0, 2, 1)
      
image_test=torch.rand((1,3,224,224))

height,width,channels=image_test.shape[-2],image_test.shape[-1],image_test.shape[1]
patch_size=16
number_patches=(height*width) // patch_size**2


patchify = PatchEmbedding(in_channels=3,
                          patch_size=16,
                          embedding_dim=768)
patched=patchify(image_test)
assert patched.shape == torch.Size([1,number_patches,patch_size**2 * channels])

#============================================CLASS + POSITIONAL EMBEDDING=========================================

batch_size = patched.shape[0]
embedding_dimension = patched.shape[-1]

class_token=nn.Parameter(torch.randn(batch_size,1,embedding_dimension),requires_grad=True)
patched_class = torch.cat((class_token, patched), dim=1)
assert patched_class.shape == torch.Size([1, 197, 768])

position_embedding = nn.Parameter(torch.randn(1,number_patches+1, embedding_dimension),requires_grad=True)
patched_class_position = patched_class + position_embedding
assert patched_class_position.shape == patched_class.shape

#============================================MULTI-HEAD SELF-ATTENTION LAYER=========================================

class MultiheadSelfAttentionBlock(nn.Module):
    def __init__(self,
                 embedding_dim:int=768,
                 num_heads:int=12,
                 attn_dropout:float=0):
        super().__init__()
        
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)
        
        # 4. Create the Multi-Head Attention (MSA) layer
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embedding_dim,
                                                    num_heads=num_heads,
                                                    dropout=attn_dropout,
                                                    batch_first=True) # does our batch dimension come first?
        
    def forward(self, x):
        x = self.layer_norm(x)
        attn_output, _ = self.multihead_attn(query=x, 
                                             key=x, 
                                             value=x,
                                             need_weights=False) 
        return attn_output
      
     
multihead_self_attention_block = MultiheadSelfAttentionBlock(embedding_dim=768,num_heads=12) 
patched_msa = multihead_self_attention_block(patch_class_position)
assert patched_msa.shape == patched_class_position.shape

#============================================LINEAR LAYERS=========================================

class LinearBlock(nn.Module):
    def __init__(self,
                 embedding_dim:int=768,
                 mlp_size:int=3072, 
                 dropout:float=0.1): 
        super().__init__()
        
        self.layer_norm = nn.LayerNorm(nembedding_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim,mlp_size),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(mlp_size, embedding_dim),
            nn.Dropout(p=dropout) 
        )
    
    def forward(self, x):
        x = self.layer_norm(x)
        x = self.mlp(x)
        return x
      
linear_block = LinearBlock(embedding_dim=768,mlp_size=3072,dropout=0.1) 
patched_linear = linear_block(patched_msa)
assert patched_linear.shape == patched_msa.shape

#============================================TRANSFORMER BLOCK=========================================

class TransformerBlock(nn.Module):
    def __init__(self,
                 embedding_dim:int=768,
                 num_heads:int=12, 
                 mlp_size:int=3072, 
                 mlp_dropout:float=0.1, 
                 attn_dropout:float=0): 
        super().__init__()

        self.msa_block = MultiheadSelfAttentionBlock(embedding_dim=embedding_dim,
                                                     num_heads=num_heads,
                                                     attn_dropout=attn_dropout)
        
        self.linear_block =  LinearBlock(embedding_dim=embedding_dim,
                                   mlp_size=mlp_size,
                                   dropout=mlp_dropout)
        
    def forward(self, x):
        x =  self.msa_block(x) + x 
        
        x = self.linear_block(x) + x 
        
        return x
      
transformer_layer = nn.TransformerBlock(d_model=768,
                                        nhead=12, 
                                        dim_feedforward=3072, 
                                        dropout=0.1,
                                        activation="gelu",
                                        batch_first=True, 
                                        norm_first=True)
transformer_output = transformer_layer(patch_class_position)

assert transformer_output.shape == patch_class_position.shape

#============================================ViT CLASS=========================================

class ViT(nn.Module):
    def __init__(self,
                 img_size:int=224,
                 in_channels:int=3, 
                 patch_size:int=16, 
                 num_transformer_layers:int=12,
                 embedding_dim:int=768,
                 mlp_size:int=3072, 
                 num_heads:int=12, 
                 attn_dropout:float=0, 
                 mlp_dropout:float=0.1,
                 embedding_dropout:float=0.1, 
                 num_classes:int=1000): 
        super().__init__()
                         
        assert img_size % patch_size == 0, f"Image size must be divisible by patch size, image size: {img_size}, patch size: {patch_size}."
        
        self.num_patches = (img_size * img_size) // patch_size**2
                 
        self.class_embedding = nn.Parameter(torch.randn(1, 1, embedding_dim),requires_grad=True)
        
        self.position_embedding = nn.Parameter(torch.randn(1, self.num_patches+1, embedding_dim),requires_grad=True)
                
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)
        
        self.patch_embedding = PatchEmbedding(in_channels=in_channels,patch_size=patch_size,embedding_dim=embedding_dim)
        
        self.transformer_encoder = nn.Sequential(*TransformerBlock(embedding_dim=embedding_dim,
                                                                   num_heads=num_heads,
                                                                   mlp_size=mlp_size,
                                                                   mlp_dropout=mlp_dropout) for _ in range(num_transformer_layers)])
       
       
        self.classifier = nn.Sequential(nn.LayerNorm(embedding_dim),nn.Linear(in_features=embedding_dim, out_features=num_classes))
    
    def forward(self, x):
        
        batch_size = x.shape[0]
        
        class_token = self.class_embedding.expand(batch_size, -1, -1) 

        x = self.patch_embedding(x)

        x = torch.cat((class_token, x), dim=1)

        x = self.position_embedding + x

        x = self.embedding_dropout(x)

        x = self.transformer_encoder(x)

        x = self.classifier(x[:, 0]) # run on each sample in a batch at 0 index

        return x
      
VisualTransformer=ViT(img_size=224,in_channels=3,patch_size=16, num_transformer_layers=12,embedding_dim=768,
        mlp_size=3072,num_heads=12, attn_dropout=0, mlp_dropout=0.1,embedding_dropout=0.1, 
        num_classes=1000)
image_batch=torch.rand((16,3,224,224))
output=VisualTransformer(image_batch)
assert output.shape == torch.Size([16,1000])

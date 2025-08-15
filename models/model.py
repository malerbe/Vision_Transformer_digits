import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    # Allows to split the image into patches
    def __init__(self,
                 img_size,
                 patch_size,
                 in_channels,
                 embed_dim):
        super().__init__()
        self.patch_size = patch_size
        
        self.proj = nn.Conv2d(in_channels=in_channels,
                              out_channels=embed_dim,
                              kernel_size=patch_size,
                              stride=patch_size)
        
        img_width, img_length = img_size[0], img_size[1]
        patch_width, patch_length = patch_size[0], patch_size[1]

        num_patches = (img_width//patch_width)*(img_length//patch_length)

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        # Le CLS Token: 
        # - Agrège l'information globale de l'image pour la classification.
        # - Sa sortie finale est utilisée pour la prédiction (ex: classification d'image).


        self.pos_embed = nn.Parameter(torch.randn(1, 1 + num_patches, embed_dim))
        # Rôle : Ajouter des informations de position aux patches (car les Transformers n'ont pas de notion d'ordre).
        # Initialisé aléatoirement et appris pendant l'entraînement (contrairement aux positional embeddings fixes en NLP).
        # 1 séquence, 1 + num_patches tokens, dimension embed_dim

    def forward(self,
                x: torch.Tensor):
        
        B = x.size(0) # Récupère le batch size

        x = self.proj(x) # Forme d'entrée: (B, C, Height, Width) --> (B, embed_dim, Height//Patchsize, Width//Patchsize)
        # Applique la convolution self.proj (définie dans __init__) pour découper l'image en patches et les projeter en embeddings.

        x = x.flatten(2) # (B, (B, N+1, E) = (32, 197, 768), N) où N = (H//patch_size) * (W//patch_size) (nombre total de patches)
        x = x.transpose(1, 2) # (B, N, embed_dim)

        cls_tokens = self.cls_token.expand(B, -1, -1) # expand(B, -1, -1) le répète pour chaque image du batch → forme (B, 1, E).
        x = torch.cat((cls_tokens, x), dim=1) # (B, N+1, embed_dim) 

        x = x + self.pos_embed
        return x
    

class MLP(nn.Module): # = Feed Forward Network
    def __init__(self,
                 in_features, 
                 hidden_features,
                 drop_rate):
        super().__init__()
        self.fc1 = nn.Linear(in_features=in_features,  # fc for fully connected
                             out_features=hidden_features)
        self.fc2 = nn.Linear(in_features=hidden_features,
                             out_features=in_features)
        
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        x = self.dropout(F.gelu(self.fc1(x))) # Applique la première couche linéaire avec GELU et dropout
        x = self.dropout(self.fc2(x)) # Applique la seconde couche linéaire
        return x 
    


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, drop_rate):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=drop_rate, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_dim, drop_rate)

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x
    
class VisionTransformer(nn.Module):
    def __init__(self,
                 img_size,
                 patch_size,
                 in_channels,
                 num_classes,
                 embed_dim,
                 depth,
                 num_heads,
                 mlp_dim,
                drop_rate):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size=img_size,
                                          patch_size=patch_size,
                                          in_channels=in_channels,
                                          embed_dim=embed_dim)
         
        self.encoder = nn.Sequential(*[
            TransformerEncoderLayer(embed_dim=embed_dim,
                                    num_heads=num_heads,
                                    mlp_dim=mlp_dim,
                                    drop_rate=drop_rate)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.encoder(x)
        x = self.norm(x)

        cls_token = x[:, 0]  
        
        return self.head(cls_token)  # Classification
    
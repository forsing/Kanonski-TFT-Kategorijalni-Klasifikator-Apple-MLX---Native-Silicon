# Model V4: Kanonski TFT Kategorijalni Klasifikator (Apple MLX - Native Silicon)



import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import pandas as pd
import numpy as np
import time

# --- UČITAVANJE PODATAKA ---
def ucitaj_mlx_podatke(csv_path, prozor=200):
    df = pd.read_csv(csv_path, header=None)
    matrica = df.values.astype(np.int32)
    
    x_lista, y_lista = [], []
    for i in range(len(matrica) - prozor):
        x_lista.append(matrica[i : i + prozor])
        y_lista.append(matrica[i + prozor])
        
    return mx.array(np.array(x_lista)), mx.array(np.array(y_lista))

# --- ARHITEKTURA KANONSKOG TFT ZA MLX ---
class MLX_GLU(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.linear = nn.Linear(d_model, d_model * 2)
    def __call__(self, x):
        x = self.linear(x)
        x, gate = mx.split(x, 2, axis=-1)
        return x * mx.sigmoid(gate)

class MLX_GRN(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_model)
        self.linear_2 = nn.Linear(d_model, d_model)
        self.glu = MLX_GLU(d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    def __call__(self, x):
        residual = x
        x = nn.elu(self.linear_1(x))
        x = self.linear_2(x)
        x = self.dropout(x)
        x = self.glu(x)
        return self.layer_norm(x + residual)

class MLX_VariableSelectionNetwork(nn.Module):
    def __init__(self, num_features, d_model):
        super().__init__()
        self.num_features = num_features
        self.feature_grns = [MLX_GRN(d_model) for _ in range(num_features)]
        self.flattened_grn = MLX_GRN(num_features * d_model)
        self.weight_linear = nn.Linear(num_features * d_model, num_features)

    def __call__(self, x_list):
        processed = [self.feature_grns[i](x_list[i]) for i in range(self.num_features)]
        flat_features = mx.concatenate(processed, axis=-1)
        
        weights = self.flattened_grn(flat_features)
        weights = self.weight_linear(weights)
        weights = mx.softmax(weights, axis=-1)
        
        weights = mx.expand_dims(weights, axis=-1) 
        stacked = mx.stack(processed, axis=-2) 
        return mx.sum(weights * stacked, axis=-2)

class MLX_InterpretableMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
    def __call__(self, x):
        b, l, d_model = x.shape
        q = self.q_linear(x).reshape(b, l, self.n_heads, self.d_head).transpose(0, 2, 1, 3)
        k = self.k_linear(x).reshape(b, l, self.n_heads, self.d_head).transpose(0, 2, 1, 3)
        v = self.v_linear(x).reshape(b, l, self.n_heads, self.d_head).transpose(0, 2, 1, 3)
        scores = mx.matmul(q, k.transpose(0, 1, 3, 2)) / (self.d_head ** 0.5)
        mask = mx.triu(mx.ones((l, l)), k=1).astype(mx.bool_)
        scores = mx.where(mask, float('-inf'), scores)
        attn_weights = mx.softmax(scores, axis=-1)
        context = mx.matmul(attn_weights, v).transpose(0, 2, 1, 3).reshape(b, l, d_model)
        return self.out_linear(context)

class MLXCanonicalTFT(nn.Module):
    def __init__(self, vocab_size=40, d_model=64, n_heads=4):
        super().__init__()
        self.embeddings = [nn.Embedding(vocab_size, d_model) for _ in range(7)]
        self.vsn = MLX_VariableSelectionNetwork(num_features=7, d_model=d_model)
        self.lstm = nn.LSTM(d_model, d_model)
        self.attn = MLX_InterpretableMultiHeadAttention(d_model, n_heads)
        self.post_attn_grn = MLX_GRN(d_model)
        self.output_heads = [nn.Linear(d_model, vocab_size) for _ in range(7)]

    def __call__(self, x):
        x_features = [self.embeddings[i](x[:, :, i]) for i in range(7)]
        vsn_out = self.vsn(x_features) 
        lstm_out, _ = self.lstm(vsn_out)
        attn_out = self.attn(lstm_out)
        final_state = self.post_attn_grn(attn_out)[:, -1, :]
        izlazi = [head(final_state) for head in self.output_heads]
        return mx.stack(izlazi, axis=1)

# --- FUNKCIJA GUBITKA ---
def loss_fn(model, x, y):
    izlaz = model(x)
    ukupni_loss = 0.0
    for i in range(7):
        ukupni_loss += mx.mean(nn.losses.cross_entropy(izlaz[:, i, :], y[:, i]))
    return ukupni_loss

# --- TRENING PETLJA ---
def treniraj_v2():
    csv_putanja = "/Users/4c/Desktop/GHQ/data/loto7_4682_k72_loto_2963.csv"
    X, Y = ucitaj_mlx_podatke(csv_putanja, prozor=100)
    
    model = MLXCanonicalTFT()
    mx.eval(model.parameters())
    
    optimizer = optim.Adam(learning_rate=0.001)
    loss_and_grads = nn.value_and_grad(model, loss_fn)
    
    batch_size = 32
    # ISPRAVLJENO: Uzima se indeks 0 za tačan broj uzoraka
    num_samples = X.shape[0]
    
    print("Model V2: Kanonski TFT Kategorijalni Klasifikator (Apple MLX - Native Silicon)")
    print("Trening modela je pokrenut...")
    start_time = time.time()
    
    for epoha in range(1, 1001):
        indeksi = np.random.permutation(num_samples)
        epoha_loss = 0.0
        koraci = 0
        
        for k in range(0, num_samples, batch_size):
            batch_indeksi = mx.array(indeksi[k : k + batch_size])
            x_b = X[batch_indeksi]
            y_b = Y[batch_indeksi]
            
            loss, grads = loss_and_grads(model, x_b, y_b)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), loss)
            
            epoha_loss += loss.item()
            koraci += 1
            
        if epoha % 50 == 0:
            print(f"Epoha [{epoha}/1200] | Kategorijalni Gubitak: {epoha_loss/koraci:.4f}")
            
    print(f"Trening završen za: {time.time() - start_time:.2f} sekundi.")
            
    # Predikcija na osnovu zadnjih 200 redova
    df_provera = pd.read_csv(csv_putanja, header=None)
    zadnji_prozor = mx.array(df_provera.values[-200:].astype(np.int32)).unsqueeze(0)
    
    izlaz_pred = model(zadnji_prozor)
    predikcija = mx.argmax(izlaz_pred, axis=-1).squeeze(0)
    mx.eval(predikcija)
    
    print("\n==================================================")
    print(f"REZULTAT ZA FAJL {csv_putanja} (Sledeći red - MLX V4):")
    print(np.array(predikcija))
    print("==================================================")

if __name__ == "__main__":
    treniraj_v2()



"""

"""



"""
Prepoznavanje obrazaca (Pattern Recognition) 
Otkrivanje obrazaca (Pattern Discovery) 
Rudarenje obrazaca (Pattern Mining)   --->   Mamba-2

Model V1: Mamba-2 SSD Regresioni Model (PyTorch - Apple Silicon Compatible)
Model V2: Mamba-2 SSD Kategorijalni Klasifikator (Apple MLX - Native Silicon)
Model V3: Kanonski TFT Kategorijalni Klasifikator (PyTorch - Apple Silicon Compatible) 
Model V4: Kanonski TFT Kategorijalni Klasifikator (Apple MLX - Native Silicon)

Arhitektura Mamba 2 se zasniva na teoriji Structured State Space Duality (SSD). 
Mamba 2 omogućava da se proračun stanja transformiše u blokovske matrične multiplikacije, 
što je znatno lakše napisati u čistom Python-u/PyTorch-u. 
"""



"""
Optimalni odnosa između dužine istorijskog prozora i broja epoha za bazu podataka. 
Cilj je balans: dovoljno velik prozor da Mamba-2 uhvati cikluse, 
ali dovoljno primera za trening da model ne upadne u hiper-podešavanje (overfitting).

Evo optimalnih vrednosti za oba modela na osnovu količine podataka u tri CSV fajla, 
kako bi se sprečio overfitting (prenaučenost) i maksimalno iskoristila dužina istorije: 

Model V1,V3: PyTorch (Kraći prozor, brža konvergencija)
Za 4682 reda: Prozor: 40 | Epohe: 150 
Za 2963 reda: Prozor: 30 | Epohe: 120 
Za 1719 reda: Prozor: 20 | Epohe: 100  

Model V2,V4: Apple MLX (Širi prozor, dublja istorija)
Za 4682 reda: Prozor: 200 | Epohe: 1200 
Za 2963 reda: Prozor: 100 | Epohe: 1000 
Za 1719 reda: Prozor:  50 | Epohe: 400 
"""



"""
Mamba / S4 (State Space Models - SSM) 
Najnovija generacija AI arhitektura koja u mnogim zadacima predviđanja sekvenci nadmašuje čak i Transformere. 
Mamba ima linearno skaliranje i koristi selektivni mehanizam stanja. 
Za razliku od standardnih modela koji se muče sa dugoročnim zavisnostima u brojevima, 
Mamba može da kompresuje celu istoriju u jedno kompaktno "stanje" i precizno uoči ako se u CSV fajlu krije složen, 
visokodimenzionalni matematički algoritam ili generator.


Mamba / S4 (State Space Models) je arhitektonski napredniji i teoretski moćniji model od TFT-a za pronalaženje dubokih zakonitosti u dugim nizovima. 
Mamba je dizajnirana upravo da reši najveću manu starijih modela: sposobnost da filtrira nevažne podatke i zadrži savršen matematički fokus na ključnim promenama kroz vreme, bez gubitka memorije. 
Kroz svoj selektivni mehanizam stanja (Selective State Space), Mamba će pokušati da mapira skrivenu funkciju koja generiše ove brojeve i izračuna tačne vrednosti za sledećih 7 brojeva (next red).



Mamba / S4 ima suštinske prednosti koje direktno utiču na pronalaženje dubokih zakonitosti u loto kombinacijama: 

Efektivni kontekst nad dugom istorijom: 
Mamba koristi linearni selective scan mehanizam koji kompresuje celu istoriju CSV redova u jedno skriveno stanje konstantne veličine. 
TFT se oslanja na pažnju (Attention) koja ima kvadratnu složenost i gubi stabilnost kada prozor postane preveliki. 

Neprekidno modelovanje vremena (Continuous-time SSM): 
Mamba kroz diskretizaciju (Delta) uči skriveni kontinuum i dinamiku sistema. 
Ona tretira vaš CSV kao kontinualni signal koji se razvija kroz vreme, 
što joj omogućava da uoči duboke, ciklične i skrivene repetitivne obrasce koje TFT-ovi statični prozori promašuju. 

Selekcija informacija kroz vreme: 
Mamba filtrira nevažne šumove u svakom koraku sekvence. 
Za razliku od TFT-a koji pokušava da odjednom izvaže uticaj svih kolona u fiksnom prozoru, 
Mamba dinamički odlučuje šta iz prethodnih izvlačenja treba trajno zapamtiti, a šta odbaciti.


Model koristi Embedding sloj veličine 40 (za brojeve 1-39, gde je 0 rezervisana za mapiranje) 
i višeslojnu kauzalnu strukturu sa mehanizmom selektivnog stanja i rezidualnim vezama.
Embedding sloj: 
Brojeve od 1 do 39 ne posmatra kao proste cifre, već kreira "guste vektore" (d_model=256). 
Na taj način model uči skriveni kontekst (npr. kako se broj 7 ponaša kada je na prvoj poziciji u odnosu na to kada je na trećoj poziciji). 

Python kod za pripremu i treniranje modela
učitava csv, priprema podatke metodom kliznog prozora (gledajući istoriju da bi predvideo sledeći red) i trenira model visoke moći.


SSM/Mamba princip duboke kompresije: 
Za razliku od klasičnih modela, unutrašnji slojevi (SSMResidualBlock) vrše ne-linearnu projekciju podataka u prostor od 512 dimenzija (d_state). 
To omogućava računaru da zadrži matematičku strukturu informacija kroz ceo niz od CSV koraka.

Automatsko sortiranje na izlazu: 
Kod na samom kraju uzima sirove matematičke vrednosti modela, osigurava da ostanu u opsegu 1-39, 
uklanja eventualne duplikate i sortira ih od najmanjeg do najvećeg, prateći strukturu prethodnih redova.

 
Kada se pokrene skriptu, ona prođe kroz svih CSV redova da bi naučila pravila. 
Kada se trening završi, model u memoriji drži skriveno stanje sistema. 
Funkcija model(poslednji_prozor) uzima sam kraj CSV fajla (istoriju neposredno pre next koraka) 
i na osnovu svega što je naučila generiše potpuno novih 7 brojeva. 
Kada se pokrene kod u terminalu, na samom dnu se dobije jasan ispis. 
"""



"""
Mamba, odnosno Selective State Space Model (SSM) — preciznije novija arhitektura Mamba-2.
Za sekvencijalno predviđanje na CSV podacima glavni model bi bio:
Mamba-2 regresioni model za sekvence skupova
Svako izvlačenje kodira se kao binarni vektor od 39 vrednosti, 
Mamba-2 obrađuje hronološki niz prethodnih izvlačenja, 
a regresiona glava daje kontinuirani skor za svih 39 brojeva. 
Sedam najvećih skorova čini NEXT.


Da bi kod bio napisan u čistoj Mamba-2 arhitekturi, 
on bi morao da koristi zvaničnu mamba_ssm biblioteku i njene specifične CUDA operatore, 
što zahteva Linux operativni sistem i grafičku kartu (GPU).
"""

from tokenize import Double
import torch
from torch import optim
from torch.nn import functional as F
from models import Discriminator, FaderNetwork
from argparse import ArgumentParser
import torchvision
from tqdm import tqdm
import matplotlib.pyplot as plt
from pyutils import show, SkyFinderDataset, CelebADataset
from torchvision.utils import make_grid
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #use GPU if available

parser = ArgumentParser(description = "customize training")
parser.add_argument('--disc_schedule', '-ds', default = '0.000001')
parser.add_argument('--fader_lr', '-f', default = '0.0002')
parser.add_argument('--disc_lr', '-d', default = '0.0002')
parser.add_argument('--latent_space_dim', default =256)
parser.add_argument('--in_channel', default=3)
parser.add_argument('--attr_dim', default =40)
parser.add_argument('--print_every', default =10)
parser.add_argument('--data', default = 'mnist')
args = parser.parse_args()


# Load Data

train_dataset = None
if args.data == "mnist":
    train_dataset = torchvision.datasets.MNIST(
    root="../mnist/",
    train=True,
    download=True,
    transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]))
elif args.data == "skyfinder":
    train_dataset = SkyFinderDataset(
        "skyfinder/complete_table_with_mcr.csv",
        "skyfinder/data/10066/",
        ["Filename", "night"],
        transform=torchvision.transforms.Compose(
            [torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((256,256)),
            torchvision.transforms.ToTensor()])
        )
elif args.data == "celeba":
    train_dataset = CelebADataset(
        "list_attr_celeba.txt",
        "celeba_small/",
        transform=torchvision.transforms.Compose(
            [torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((256,256)),
            torchvision.transforms.ToTensor()])
        )
else:
    train_dataset = torchvision.datasets.ImageFolder(
        root="../" + args.data,
        transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]))

train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=32, 
        num_workers=0,
        shuffle=True
    )

# Models
fader = FaderNetwork(args.latent_space_dim, args.in_channel, args.attr_dim)
disc = Discriminator(args.latent_space_dim)

#TRAIN/TEST

fader_optim = optim.Adam(list(fader.encoder.parameters()) + list(fader.dec_layers.parameters()), lr=float(args.fader_lr), betas=(0.5,0.999))
disc_optim = optim.Adam(disc.model.parameters(), lr=float(args.disc_lr), betas=(0.5,0.999))

def attr_loss(target, pred):
    loss = -target * torch.log(pred) - (1- target) * torch.log(1 - pred)
   
    return torch.sum(loss) / (40)

def train(epoch):
    fader.train() #set to eval mode
    disc.train()

    sum_disc_loss = 0
    sum_disc_acc = 0
    sum_rec_loss = 0
    sum_fader_loss = 0
    disc_weight = 0
    disc_weight = epoch*float(args.disc_schedule)  #Use as a knob for tuning the weight of the discriminator in the loss function


    for batch, labels in tqdm(train_loader, desc="Epoch {}".format(epoch)):
        data = batch.to(device)
        #labels = torch.tensor(labels, dtype=torch.float).to(device)
        batch_size = labels.shape[0]
        labels = labels.view(batch_size,40)
        labels[labels < 0] = 0
        disc_optim.zero_grad()

        if data.shape[0] <= 1:
            break
        
        # Encode data
        z = fader.encode(data)

        # Prepare attributes
        batch_size = len(labels)
        hot_digits = torch.zeros((batch_size, 40, 4, 4)).to(device)
        for i, digit in enumerate(labels):
            digit[digit < 0] = 0
            digit = digit.view(40, 1, 1)
            digit = torch.cat([digit, digit], dim=1)
            digit = torch.cat([digit, digit], dim=2)
            digit = torch.cat([digit, digit], dim=1)
            digit = torch.cat([digit, digit], dim=2)
            
            hot_digits[i,:,:,:] = digit
        
        # Train discriminator
        label_probs = disc(z)
        #disc_loss = F.cross_entropy(label_probs, labels, reduction='mean')
        disc_loss = attr_loss(labels, label_probs)
        #disc_loss = F.mse_loss(label_probs, labels)
        sum_disc_loss += disc_loss.item()

        disc_loss.backward()
        disc_optim.step()

        # Compute discriminator accuracy
        #disc_pred = torch.argmax(label_probs, 1)
        #disc_acc = torch.sum(disc_pred == labels)
        label_probs[label_probs < 0.5] = 0
        label_probs[label_probs >= 0.5] = 1
        disc_acc = torch.sum(label_probs == labels) / 40
        sum_disc_acc += disc_acc.item()
        
        
        # Train Fader
        fader_optim.zero_grad()
        z = fader.encode(data)
        
        # Invariance of latent space from new disc
        label_probs = disc(z)
        
        # Reconstruction
        reconsts = fader.decode(z, hot_digits)
        rec_loss = F.mse_loss(reconsts, data, reduction='mean')
        sum_rec_loss += rec_loss.item()
        fader_loss = rec_loss - disc_weight * attr_loss(1 - labels, label_probs)

        fader_loss.backward()
        fader_optim.step()
        
        sum_fader_loss += fader_loss.item()        
        
    train_size = len(train_loader.dataset)

    print('\nDisc Weight: {:.8f} | Fader Loss: {:.8f} | Rec Loss: {:.8f} | Disc Loss, Acc: {:.8}, {:.8f}'
          .format(disc_weight, sum_fader_loss/train_size, sum_rec_loss/train_size, 
        sum_disc_loss/train_size, sum_disc_acc/train_size), flush=True)
    
    return sum_rec_loss/train_size, sum_disc_acc/train_size, disc_weight

def test(epoch):
    fader.eval() #set to eval mode
    disc.eval()
    rec_losses = 0
    disc_losses = 0
    disc_accs = 0
    
    with torch.no_grad():
        for batch, labels in train_loader:
            # Encode batch
            labels = torch.tensor(labels, dtype=torch.float).to(device)
            data_batch = batch.to(device)
            z = fader.encode(data_batch)

            # Prepare attributes
            batch_size = len(labels)
            hot_digits = torch.zeros((batch_size, 40, 4, 4)).to(device)
            for i, digit in enumerate(labels):
                digit[digit < 0] = 0
                digit = digit.view(40, 1, 1)
                digit = torch.cat([digit, digit], dim=1)
                digit = torch.cat([digit, digit], dim=2)
                digit = torch.cat([digit, digit], dim=1)
                digit = torch.cat([digit, digit], dim=2)
            
                hot_digits[i,:,:,:] = digit

            # Reconstruct
            # label_probs = disc(z)
            # disc_loss = F.cross_entropy(label_probs, labels, reduction='mean')

            reconsts = fader(data_batch, hot_digits)
            # rec_loss = F.mse_loss(reconsts, data_batch, reduction='mean')

            # disc_pred = torch.argmax(label_probs, 1)
            # disc_acc = torch.sum(disc_pred == labels)   

            # disc_losses += disc_loss.item()
            # rec_losses += rec_loss.item()
            # disc_accs += disc_acc.item()

            '''
            KEYS

            0 - original
            1 - reconstruction with original attributes
            2 - reconstruction with modified attributes

            '''

            plt.clf()

            show(make_grid(data_batch.detach().cpu()), 'Epoch {} Original'.format(epoch),epoch,"img")
            show(make_grid(reconsts), 'Epoch {} Reconst with Orig Attr'.format(epoch),epoch,"orig")

            hot_digits[:,15,:,:] = 1

            fader_reconst = fader(data_batch, hot_digits).cpu()
            show(make_grid(fader_reconst), 'Epoch {} Reconst With Attr Sunglasses'.format(epoch),epoch,"mod")
            break

        # print('Test Rec Loss: {:.8f}'.format(rec_losses / len(train_loader.dataset)))
        # print('Test disc Loss: {:.8f}'.format(disc_losses / len(train_loader.dataset)))
        # print('Test disc accs: {:.8f}'.format(disc_accs / len(train_loader.dataset)))

epochs = 1001 

recs, accs, disc_wts = [], [], []
for epoch in range(epochs):
    rec_loss, disc_acc, disc_wt = train(epoch)
    recs.append(rec_loss)
    accs.append(disc_acc)
    disc_wts.append(disc_wt)
    
    if epoch % 10 == 0:
        test(epoch)
        plt.figure(figsize=(9,3))
        plt.subplot(1,3,1)
        plt.title('Disc Weight')
        plt.plot(disc_wts)
        plt.subplot(1,3,2)
        plt.title('Reconst Loss')
        plt.plot(recs)
        plt.subplot(1,3,3)
        plt.title('Disc Acc')
        plt.plot(accs)
        plt.savefig('results/losses'+str(epoch)+'.png')
        torch.save(fader.state_dict(), 'results/fader'+str(epoch)+'.pt')


plt.figure(figsize=(9,3))
plt.subplot(1,3,1)
plt.title('Disc Weight')
plt.plot(disc_wts)
plt.subplot(1,3,2)
plt.title('Reconst Loss')
plt.plot(recs)
plt.subplot(1,3,3)
plt.title('Disc Acc')
plt.plot(accs)

plt.savefig('plots.png')


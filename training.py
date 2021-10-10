import numpy as np
import torch
from torch import nn
import pandas as pd
from torch.utils.data import DataLoader
from sklearn import metrics
from torch.optim.lr_scheduler import StepLR

#загружаем архитектуру модели для дальнейшего обучения
from model_structure import model

#Загружаем данные в DataFrame
data = pd.read_csv('data.csv')
num_samples = len(data)

# Создаём класс для загрузки данных в Dataloader pytorch
class Dataset(torch.utils.data.Dataset):
  def __init__(self, image, labels):
        'Initialization'
        self.labels = labels
        self.image = image

  def __len__(self):
        return len(self.image)

  def __getitem__(self, index):
        X = self.image[index]
        X = [int(x) for x in (X.split(' '))]
        X = np.reshape(X, (48,48))
        X = torch.from_numpy(X)

        y = torch.from_numpy(self.labels[index])

        return X, y

#Создаём генератор произвольных обучающих и тренировачных выборок в виде DataLoader pytroch
data_set = Dataset(data.pixels, data[['age', 'ethnicity', 'gender']].values)
train_data, test_data = torch.utils.data.random_split(data_set, [int(num_samples*0.8), int(num_samples*0.2)], 
                                                 generator=torch.Generator().manual_seed(42))

training_generator = torch.utils.data.DataLoader(train_data, batch_size = 64, shuffle = True, drop_last = True)
validation_generator = torch.utils.data.DataLoader(test_data, batch_size = 64, shuffle = True, drop_last = True)


#Составляем функцию обучения и валидации с замедлением скорости обучения каждые 12 эпох.
train_age = []
train_eth = []
train_gen = []

test_age = []
test_eth = []
test_gen = []

train_los_age = []
train_los_eth = []
train_los_gen = []

test_los_age = []
test_los_eth = []
test_los_gen = []

def train(model, x_train, valid,  learning_rate = 0.01, num_epochs = 8):
    
    best_score = -100
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
    lr_step = StepLR(optimizer, step_size=12, gamma=0.2)
    loss_a = nn.MSELoss()
    loss_e = nn.CrossEntropyLoss()
    loss_g = nn.BCELoss()

    for epoch in range(num_epochs):
        
        total_loss = 0  
        loss_age1 = 0
        loss_eth1 = 0
        loss_gen1 = 0

        err_age_epoch = 0
        accur_eth_epoch = 0
        accur_gen_epoch = 0

        batch_counter = 0

        #Итерации обучения
        print('обучение')
        for image, labels in x_train:
            batch_counter += 1
            optimizer.zero_grad()
            model.train()
            x_gen, x_age, x_etn = model(image.unsqueeze(1).float().requires_grad_())

            #считаем и охраняем потери
            loss_age = loss_a(x_age.squeeze(), labels[:, 0].type(torch.float32))
            loss_eth = loss_e(x_etn, labels[:, 1].type(torch.long))
            loss_gen = loss_g(x_gen.float(), labels[:, 2].unsqueeze(1).type(torch.float32))
            
            total_loss +=  loss_age.item() + loss_eth.item() + loss_gen.item()

            #считам градиент потерь для каждой головы и делаем оптимизационный шаг
            loss_age.backward(retain_graph = True)
            loss_eth.backward(retain_graph = True)
            loss_gen.backward()
            optimizer.step()

            loss_age1 += loss_age.item()
            loss_eth1 += loss_eth.item()
            loss_gen1 += loss_gen.item()

            #Получаем значения предсказаний в формате integer для использования в метриках
            _, eth_output = torch.max(x_etn, dim = 1)
            gen_output = [1  if x>0.5 else 0 for x in x_gen]

            err_age_epoch +=  metrics.mean_absolute_error(labels[:, 0], x_age.int())
            accur_eth_epoch += metrics.accuracy_score(labels[:, 1], eth_output) 
            accur_gen_epoch += metrics.accuracy_score(labels[:, 2].unsqueeze(1), gen_output)

            del(x_gen, x_age, x_etn, loss_age, loss_eth, loss_gen)

        train_age.append(err_age_epoch/batch_counter)
        train_eth.append(accur_eth_epoch/batch_counter)
        train_gen.append(accur_gen_epoch/batch_counter)

        train_los_age.append(loss_age1/batch_counter)
        train_los_eth.append(loss_eth1/batch_counter)
        train_los_gen.append(loss_gen1/batch_counter)


        print(f'Эпоха {epoch+1}')
        print(f'Ошибка возраста: {err_age_epoch/batch_counter} \n \
              Точность этноса: {accur_eth_epoch/batch_counter} \n \
              Точность пола: {accur_gen_epoch/batch_counter}')
        print(f'Общие потери: {total_loss} \n \
                Потери возраст: {loss_age1/batch_counter}\n \
                Потери пол: {loss_gen1/batch_counter}\n \
                Потери этнос: {loss_eth1/batch_counter}')

        del(batch_counter, loss_age1, loss_eth1, loss_gen1)

        batch_counter = 0
        err_age_epoch = 0
        accur_eth_epoch = 0
        accur_gen_epoch = 0
        loss_age1 = 0
        loss_eth1 = 0
        loss_gen1 = 0

        #### Итерации валидации
        print('тестирование')
        with torch.no_grad():
            for image, labels in valid:
                batch_counter += 1
                model.eval()

                x_gen, x_age, x_etn = model(image.unsqueeze(1).float())
                loss_age = loss_a(x_age.squeeze(), labels[:, 0].type(torch.float32))
                loss_eth = loss_e(x_etn, labels[:, 1])
                loss_gen = loss_g(x_gen.float(), labels[:, 2].unsqueeze(1).type(torch.float32))
            
                total_loss +=  loss_age.item() + loss_eth.item() + loss_gen.item()

                loss_age1 += loss_age.item()
                loss_eth1 += loss_eth.item()
                loss_gen1 += loss_gen.item()

                _, eth_output = torch.max(x_etn, dim = 1)
                gen_output = [1  if x>0.5 else 0 for x in x_gen]

                err_age_epoch += metrics.mean_absolute_error(labels[:, 0], x_age.int())
                accur_eth_epoch += metrics.accuracy_score(labels[:, 1], eth_output)
                accur_gen_epoch += metrics.accuracy_score(labels[:, 2].unsqueeze(1), gen_output)
                
                del(x_gen, x_age, x_etn, loss_age, loss_eth, loss_gen)

            test_age.append(err_age_epoch/batch_counter)
            test_eth.append(accur_eth_epoch/batch_counter)
            test_gen.append(accur_gen_epoch/batch_counter)
                
            test_los_age.append(loss_age1/batch_counter)
            test_los_eth.append(loss_eth1/batch_counter)
            test_los_gen.append(loss_gen1/batch_counter)
                
            print(f'Ошибка возраста: {err_age_epoch/batch_counter} \n \
                        Точность этноса: {accur_eth_epoch/batch_counter} \n \
                        Точность пола: {accur_gen_epoch/batch_counter}')
            print(f'Общие потери: {total_loss} \n \
                        Потери возраст: {loss_age1/batch_counter}\n \
                        Потери пол: {loss_gen1/batch_counter}\n \
                        Потери этнос: {loss_eth1/batch_counter}')

            cur_score = (accur_eth_epoch/batch_counter + accur_gen_epoch/batch_counter) - err_age_epoch/batch_counter
            
            #Сохраняем модель с лучшими показателями метрики
            if best_score < cur_score:
                best_score = cur_score
                torch.save(model.state_dict(), f'model_epoch{epoch+1}.pth')
            
            del(batch_counter, loss_age1, loss_eth1, loss_gen1)
        
        #Изменяем скорость обучения, если прошло 12 эпох
        lr_step.step()


train(model, x_train = training_generator, valid = validation_generator, num_epochs = 30)
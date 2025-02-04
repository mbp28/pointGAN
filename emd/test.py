import torch
import torch.nn as nn
from torch.autograd import Variable

from modules.emd import EMDModule

dist =  EMDModule()

p1 = torch.rand(1,1000,6)
p2 = torch.rand(1,1000,6)
points1 = Variable(p1,requires_grad = True)
points2 = Variable(p2, requires_grad = True)
cost = dist(points1, points2)
print(cost)
loss = torch.sum(cost)
print(loss)
loss.backward()
print(points1.grad, points2.grad)


points1 = Variable(p1.cuda(), requires_grad = True)
points2 = Variable(p2.cuda(), requires_grad = True)
cost = dist(points1, points2)
print(cost)
loss = torch.sum(cost)
print(loss)
loss.backward()
print(points1.grad, points2.grad)

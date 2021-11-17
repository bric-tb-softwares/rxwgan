

from rxwgan.models.models_v1 import *

critic = Critic_v1()
gen = Generator_v1()

critic.model.save('critic.h5')
gen.model.save('generator.h5')


import FlowCompression
from FlowCompression.ClassAE import AE
from FlowCompression.ParentClass import Model


u_train, u_val, u_test = AE.preprocess(filename='Kolmogorov_Re40.0_T6000_DT001_res33.h5', nx=48)

#model = AE.create_trained(2)  # -> Comment model = AE(), and model.fit() to run pre trained
model = AE(nx=48)

model.fit(u_train, u_val)

t = model.passthrough(u_test)
# model.vorticity_energy()
perf = model.performance()
# AE.plot_all(t[0])

Model.verification(u_test)
Model.verification(t)

print(f'Absolute %: {round(perf["abs_percentage"], 3)} +- {round(perf["abs_std"], 3)}')
print(f'Squared %: {round(perf["sqr_percentage"], 3)} +- {round(perf["sqr_std"], 3)}')
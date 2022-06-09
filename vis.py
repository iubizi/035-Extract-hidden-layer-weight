####################
# read npz
####################

import numpy as np

doc = np.load('extract_output.npz')

extract_output = doc['extract_output']
y_test = doc['y_test']

print('extract_output.shape =', extract_output.shape)
print('y_test.shape =', y_test.shape)
print()

####################
# tsne
####################

from sklearn.manifold import TSNE

tsne = TSNE( n_components=2,
             learning_rate='auto',
             init='random' )
x_tsne = tsne.fit_transform(extract_output)

####################
# visualization
####################

import matplotlib.pyplot as plt

# different class have different color
scatter = plt.scatter( x_tsne[:, 0], x_tsne[:, 1],
                       c=y_test, cmap='jet', # alpha=0.5,
                       marker='.')
# auto legend
plt.legend( *scatter.legend_elements(prop='colors'),
            title='mnist', loc='best' )

plt.title('visualization')
plt.tight_layout()
plt.show()

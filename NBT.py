import numpy as np
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
class NBT():
    """
    Nyström Basis Transfer Service Class

    Published in:
    Christoph Raab, Frank-Michael Schleif,
    Transfer learning extensions for the probabilistic classification vector machine,Neurocomputing,2019,
    https://doi.org/10.1016/j.neucom.2019.09.104.

    Functions
    ----------
    nys_basis_transfer: Transfer Basis from Target to Source Domain.
    data_augmentation: Augmentation of data by removing or upsampling of source data

    Examples
    --------
    >>> #Imports
    >>> import os
    >>> import scipy.io as sio
    >>> from sklearn.svm import SVC
    >>> os.chdir(os.path.dirname(os.path.abspath(__file__)))
    >>> os.chdir(os.path.join("datasets","domain_adaptation","features","OfficeCaltech"))
    >>> amazon = sio.loadmat("amazon_SURF_L10.mat")
    >>> X = preprocessing.scale(np.asarray(amazon["fts"]))
    >>> Yt = np.asarray(amazon["labels"])
    >>>
    >>> dslr = sio.loadmat("dslr_SURF_L10.mat")
    >>>
    >>> Z = preprocessing.scale(np.asarray(dslr["fts"]))
    >>> Ys = np.asarray(dslr["labels"])
    >>>
    >>> clf = SVC(gamma=1,C=10)
    >>> clf.fit(Z,Ys)
    >>> print("SVM: "+str(clf.score(X,Yt)))
    >>>
    >>> nbt = NBT()
    >>> Ys,Z = nbt.data_augmentation(Z,X.shape[0],Ys)
    >>> X,Z = nbt.nys_basis_transfer(X,Z,Ys.flatten(),landmarks=100)
    >>> clf = SVC(gamma=1,C=10)
    >>> clf.fit(Z,Ys)
    >>> print("SVM + NBT: "+str(clf.score(X,Yt)))
    """

    def __init__(self,landmarks=10):
        self.landmarks = landmarks
        pass

    def nys_basis_transfer(self,X,Z,Ys=None):
        """
        Nyström Basis Transfer
        Transfers Basis of X to Z obtained by Nyström SVD
        Implicit dimensionality reduction
        Applications in domain adaptation or transfer learning
        Parameters.
        Note target,source are order sensitiv.
        ----------
        X : Target Matrix, where classifier is trained on
        Z : Source Matrix, where classifier is trained on
        Ys: Source data label, if none, classwise sampling is not applied.
        landmarks : Positive integer as number of landmarks

        Returns
        ----------
        X : Reduced Target Matrix
        Z : Reduced approximated Source Matrix

        """
        if type(X) is not np.ndarray or type(Z) is not np.ndarray:
            raise ValueError("Numpy Arrays must be given!")
        if type(self.landmarks ) is not int or self.landmarks  < 1:
            raise ValueError("Positive integer number must given!")
        landmarks = np.min([X.shape[0]]+[Z.shape[0]]+[self.landmarks ])
        max_idx = np.min(list(X.shape)+list(Z.shape))
        idx = np.random.randint(0,max_idx-1,landmarks)
        A = X[np.ix_(idx,idx)]
        # B = X[0:landmarks,landmarks:]
        F = X[landmarks:,0:landmarks]
        #C = X[landmarks:,landmarks:]
        U, S, H = np.linalg.svd(A, full_matrices=True)
        S = np.diag(S)

        U_k = np.concatenate([U,(F @H )@np.linalg.pinv(S)])
        #V_k = np.concatenate([H, np.matmul(np.matmul(B.T,U),np.linalg.pinv(S))])
        X = U_k @S

        if type(Ys) is np.ndarray:
            A = self.classwise_sampling(Z,Ys,landmarks)
        else:
            A = Z[np.ix_(idx,idx)]

        D = np.linalg.svd(A, full_matrices=True,compute_uv=False)
        Z = U_k @ np.diag(D)
        return preprocessing.scale(X),preprocessing.scale(Z)

    def classwise_sampling(self,X,Y,n_landmarks):

        A = []
        classes = np.unique(Y)
        c_classes = classes.size
        samples_per_class = int(n_landmarks / c_classes)
        for c in classes:
            class_data = X[np.where(c == Y)]

            if samples_per_class > class_data.shape[0]:
                A = A+list(class_data)
            else:
                A = A+list(class_data[np.random.randint(0,class_data.shape[0],samples_per_class)])

        return np.array(A)

    def basis_transfer(self,X,Z):
        """
         Basis Transfer
        Transfers Basis of X to Z obtained by Nyström SVD
        Applications in domain adaptation or transfer learning
        Parameters.
        Note target,source are order sensitiv.
        ----------
        X : Target Matrix, where classifier is trained on
        Z : Source Matrix, where classifier is trained on

        Returns
        ----------
        Z : Transferred Source Matrix
        """
        L,S,R = np.linalg.svd(X,full_matrices=False);
        D = np.linalg.svd(Z,compute_uv=False,full_matrices=False)
        return L @ np.diag(D) @ R

    def fit_predict(self, Xs, Ys, Xt, Yt):
        '''
        Fit and use 1NN to classify
        :param Xs: ns * n_feature, source feature
        :param Ys: ns * 1, source label
        :param Xt: nt * n_feature, target feature
        :param Yt: nt * 1, target label
        :return: Accuracy, predicted labels of target domain, and G
        '''
        clf = KNeighborsClassifier(n_neighbors=1)
        clf.fit(Xs, Ys.ravel())
        y_pred = clf.predict(Xt)
        acc = np.mean(y_pred == Yt.ravel())
        return acc, y_pred


    def data_augmentation(self,Z,required_size,Y):
        """
        Data Augmentation
        Upsampling if Z smaller as required_size via multivariate gaussian mixture
        Downsampling if Z greater as required_size via uniform removal

        Note both are class-wise with goal to harmonize class counts
        ----------
        Z : Matrix, where classifier is trained on
        required_size : Size to which Z is reduced or extended
        Y : Label vector, which is reduced or extended like Z

        Returns
        ----------
        X : Augmented Z
        Z : Augmented Y

        """
        if type(Z) is not np.ndarray or type(required_size) is not int or type(Y) is not np.ndarray:
            raise ValueError("Numpy Arrays must be given!")
        if Z.shape[0] == required_size:
            return Y,Z
        
        _, idx = np.unique(Y, return_index=True)
        C = Y[np.sort(idx)].flatten().tolist()
        size_c = len(C)
        if Z.shape[0] < required_size:
            print("Source smaller target")
            data = np.empty((0,Z.shape[1]))
            label = np.empty((0,1))
            diff = required_size - Z.shape[0]
            sample_size = int(np.floor(diff/size_c))
            for c in C:
                #indexes = np.where(Y[Y==c])
                indexes =  np.where(Y==c)
                class_data = Z[indexes,:][0]
                m = np.mean(class_data,0) 
                sd = np.var(class_data,0)
                sample_size = sample_size if c !=C[-1] else sample_size+np.mod(diff,size_c)
                augmentation_data =np.vstack([np.random.normal(m, sd, size=len(m)) for i in range(sample_size)])
                data =np.concatenate([data,class_data,augmentation_data])
                label = np.concatenate([label,np.ones((class_data.shape[0]+sample_size,1))*c])
            
        if Z.shape[0] > required_size:
            print("Source greater target")
            data = np.empty((0,Z.shape[1]))
            label = np.empty((0,1))
            sample_size = int(np.floor(required_size/size_c))
            for c in C:
                indexes = np.where(Y[Y==c])[0]
                class_data = Z[indexes,:]
                if len(indexes) > sample_size:
                    sample_size = sample_size if c !=C[-1] else np.abs(data.shape[0]-required_size)
                    y = np.random.choice(class_data.shape[0],sample_size)
                    class_data = class_data[y,:]
                data =np.concatenate([data,class_data])
                label = np.concatenate([label,np.ones((class_data.shape[0],1))*c])
        Z = data
        Y = label
        return Y,Z

if __name__ == "__main__":

    import os
    import scipy.io as sio
    from sklearn.svm import SVC

    # Load and preprocessing of data. Note normalization to N(0,1) is necessary.
    os.chdir("datasets/domain_adaptation/OfficeCaltech/features/Surf")
    amazon = sio.loadmat("amazon_SURF_L10.mat")
    X = preprocessing.scale(np.asarray(amazon["fts"]))
    Yt = np.asarray(amazon["labels"])

    dslr = sio.loadmat("dslr_SURF_L10.mat")

    Z = preprocessing.scale(np.asarray(dslr["fts"]))
    Ys = np.asarray(dslr["labels"])

    # Applying SVM without transfer learning. Accuracy should be about 10%
    clf = SVC(gamma=1,C=10)
    clf.fit(Z,Ys)
    print("SVM without transfer "+str(clf.score(X,Yt)))

    # Beginning of NBT. Accuracy of SVM + NBT should be about 90%
    nbt = NBT(landmarks=100)
    # Data augmentation is necessary if Z and X have different shapes.
    Ys,Z = nbt.data_augmentation(Z,X.shape[0],Ys)
    X,Z = nbt.nys_basis_transfer(X,Z,Ys.flatten())
    clf = SVC(gamma=1,C=10)
    clf.fit(Z,Ys)
    print("SVM + NBT: "+str(clf.score(X,Yt)))
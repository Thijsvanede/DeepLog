import numpy as np

class LabelEncoder(object):

    def __init__(self, update=False):
        """Encode target labels with value between 0 and n_classes-1."""
        # Initialise encoding
        self.encoding = dict()
        # Set update
        self.update = update

    def fit(self, y):
        """Fit label encoder

            Parameters
            ----------
            y : array-like of shape=(n_samples,)
                Target values

            Returns
            -------
            self : LabelEncoder
                returns an instance of self
            """
        # Get unique values
        y = np.unique(y)
        # Encode labels
        encoding = {x: i+len(self.encoding) for i, x in enumerate(set(y) - set(self.encoding))}

        # Set or update self.encoding
        if self.update:
            self.encoding.update(encoding)
        else:
            self.encoding = encoding
        # Return self
        return self

    def transform(self, y):
        """Transform labels to normalized encoding

            Parameters
            ----------
            y : array-like of shape=(n_samples,)
                Target values

            Returns
            -------
            y : array-like of shape=(n_samples,)
                Encoded values
            """
        # Encode labels and return
        return np.vectorize(lambda x: self.encoding.get(x))(y)

    def fit_transform(self, y):
        """Fit label encoder and return encoded labels

            Parameters
            ----------
            y : array-like of shape=(n_samples,)
                Target values

            Returns
            -------
            y : array-like of shape=(n_samples,)
                Encoded values
            """
        return self.fit(y).transform(y)

    def inverse_transform(self, y):
        """Transform labels to normalized encoding

            Parameters
            ----------
            y : array-like of shape=(n_samples,)
                Target values

            Returns
            -------
            y : array-like of shape=(n_samples,)
                Encoded values
            """
        # Get inverse encoding
        inverse_encoding = {v: k for k, v in self.encoding.items()}
        # Decode labels and return
        return np.asarray([inverse_encoding.get(x) for x in y])


if __name__ == "__main__":
    le = LabelEncoder(update=True)
    a = np.asarray([2, 3, 3, 4, 5, 5])
    b = np.asarray([0, 1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1, 0])

    print("                  A :", a)
    print("                  B :", b)
    print()
    print("    fit_transform(A):", le.fit_transform(a))
    print("        transform(A):", le.transform(a))
    print("        transform(B):", le.transform(b))
    print("    fit_transform(B):", le.fit_transform(b))
    print("        transform(A):", le.transform(a))
    print("    fit_transform(A):", le.fit_transform(a))
    print("        transform(B):", le.transform(b))
    print()
    print("inverse_transform(A):", le.inverse_transform(le.transform(a)))
    print("inverse_transform(B):", le.inverse_transform(le.transform(b)))

class IdentityHash:
    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other

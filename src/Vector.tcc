template <typename T>
Vec<T>::Vec()
{
    rows = 0;
    columns = 0;
}

template <typename T>
Vec<T> Vec<T>::operator+(Vec<T> rhs)
{
    Vec<T> r;
    r.rows = rows;
    r.columns = columns;
    for(unsigned i = 0; i < vector.size(); ++i)
        r.vector.push_back(vector[i] + rhs.vector.at(i));
    return r;
}

template <typename T>
Vec<T> Vec<T>::operator*(Vec<T> rhs)
{
    Vec<T> r;
    r.rows = rows;
    r.columns = columns;
    for(unsigned i = 0; i < vector.size(); ++i)
        r.vector.push_back(vector.at(i) * rhs.vector.at(i));
    return r;
}

template <typename T>
Vec<T> Vec<T>::operator-(Vec<T> rhs)
{
    Vec<T> r;
    r.rows = rows;
    r.columns = columns;
    for(unsigned i = 0; i < vector.size(); ++i)
        r.vector.push_back(vector[i] - rhs.vector.at(i));
    return r;
}

template <typename T>
Vec<T> Vec<T>::operator/(Vec<T> rhs)
{
    Vec<T> r;
    r.rows = rows;
    r.columns = columns;
    for(unsigned i = 0; i < vector.size(); ++i)
        r.vector.push_back(vector[i] / rhs.vector.at(i));
    return r;
}

template <typename T>
T Vec<T>::sum()
{
    T s = 0;
    for(unsigned i = 0; i < vector.size(); ++i)
    {
        s += vector[i];
    }

    return s;
}

template <typename T>
Mat<T> Vec<T>::toMatrix()
{
    Mat<T> temp(rows, 1, 0);
    temp.matrix = vector;
    return temp;
}

template <typename T>
Mat<T> Vec<T>::transpose()
{
    Mat<T> temp;
    temp.rows = columns;
    temp.columns = rows;
    for(unsigned i = 0; i < vector.size(); ++i)
        temp.matrix.push_back(vector[i]);
    return temp;
}

template <typename T>
template <typename Functor>
Vec<T> Vec<T>::map(Functor f)
{
    Vec<T> temp;
    temp.vector = this->vector;
    temp.rows = this->rows;
    temp.columns = this->columns;
    for(unsigned i = 0; i < vector.size(); ++i)
        f(vector[i]);

    return temp;
}

template <typename T>
void Vec<T>::addColumns(int numColumns, T value)
{
    for(int i = 0; i < numColumns; ++i)
    {
        int elem = 0;
        int j = 0;
        do
        {
            vector.insert(vector.begin()+(j * columns)+elem, value);
            ++elem;
            ++j;
        }while(j < rows);
        ++columns;
    }
}

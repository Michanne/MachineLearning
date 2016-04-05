template <typename T>
Mat<T>::Mat()
{
    rows = 0;
    columns = 0;
}

template <typename T>
Mat<T>::Mat(int rows, int columns, T value)
{
    this->rows = rows;
    this->columns = columns;
    for(int i = 0; i < rows*columns; ++i)
        matrix.push_back(value);
}

template <typename T>
template <size_t r, size_t c>
Mat<T>::Mat(T(&values)[r][c])
{
    this->rows = r;
    this->columns = c;
    for(int i = 0; i < rows; ++i)
        for(int j = 0; j < columns; ++j)
            matrix.push_back(values[i][j]);
}

template <typename T>
Mat<T>::Mat(std::initializer_list<std::initializer_list<T>> values)
{
    this->rows = values.size();
    int c = 0;
    for(auto i : values)
    {
        for(auto j : i)
        {
            ++c;
            matrix.push_back(j);
        }

        this->columns = c;
    }
}

template <typename T>
Mat<T> Mat<T>::operator*(Mat<T> rhs)
{
    Mat<T> temp;

    if(rhs.rows == columns)
    {
        temp.rows = rows;
        temp.columns = rhs.columns;
        for(int i = 0; i < rows; ++i)
        {
            for(int p = 0; p < rhs.columns; ++p)
            {
                T sum = 0;

                for(int j = 0; j < rhs.rows; ++j)
                {
                    sum += matrix.at((i * columns) + j) * rhs.matrix.at((j * rhs.columns) + p);
                }

                temp.matrix.push_back(sum);
            }
        }
    }

    return temp;
}

template <typename T>
Mat<T> Mat<T>::operator*(Vec<T> rhs)
{
    Mat<T> temp;

    if(rhs.rows == columns)
    {
        temp.rows = rows;
        temp.columns = rhs.columns;
        for(int i = 0; i < rows; ++i)
        {
            for(int p = 0; p < rhs.columns; ++p)
            {
                T sum = 0;

                for(int j = 0; j < rhs.rows; ++j)
                {
                    sum += matrix.at((i * columns) + j) * rhs.vector.at((j * rhs.columns) + i);
                }

                temp.matrix.push_back(sum);
            }
        }
    }

    return temp;
}

template <typename T>
Mat<T> Mat<T>::operator-(Mat<T> rhs)
{
    Mat<T> temp;
    if(rhs.rows  == rows && rhs.columns == columns)
    {
        temp.rows = rows;
        temp.columns = columns;
        for(int i = 0; i < rows*columns; ++i)
            temp.matrix.push_back(matrix.at(i) - rhs.matrix.at(i));
    }

    return temp;
}

template <typename T>
Mat<T> Mat<T>::operator+(Mat<T> rhs)
{
    Mat<T> temp;
    if(rhs.rows  == rows && rhs.columns == columns)
    {
        temp.rows = rows;
        temp.columns = columns;
        for(int i = 0; i < rows*columns; ++i)
            temp.matrix.push_back(matrix.at(i) + rhs.matrix.at(i));
    }

    return temp;
}

template <typename T>
template <typename Functor>
Mat<T> Mat<T>::map(Functor f)
{
    Mat<T> temp;
    temp.matrix = this->matrix;
    temp.rows = this->rows;
    temp.columns = this->columns;
    for(unsigned i = 0; i < matrix.size(); ++i)
        f(temp.matrix[i]);

    return temp;
}

template <typename T>
Mat<T> Mat<T>::flatten()
{
    Mat<T> temp(1, columns, 0);
    for(int j = 0; j < columns; ++j)
    {
        T sum = 0;
        for(int i = 0; i < rows; ++i)
            sum += matrix.at((i * columns) + j);
        temp.matrix.at(j) = sum;
    }

    return temp;
}

template <typename T>
Mat<T> Mat<T>::transpose()
{
    Mat<T> temp;
    temp.rows = columns;
    temp.columns = rows;
    for(int j = 0; j < columns; ++j)
        for(int i = 0; i < rows; ++i)
            temp.matrix.push_back(matrix.at((i*columns)+j));

    return temp;
}

template <typename T>
void Mat<T>::addColumns(int numColumns, T value)
{
    for(int i = 0; i < numColumns; ++i)
    {
        int elem = 0;
        int j = 0;
        do
        {
            matrix.insert(matrix.begin()+(j * columns)+elem, value);
            ++elem;
            ++j;
        }while(j < rows);
        ++columns;
    }
}

template <typename T>
Mat<T> Mat<T>::getColumns(int begin, int end)
{
    Mat<T> temp;
    temp.rows = rows;
    temp.columns = end - begin;
    for(int i = 0; i < rows; ++i)
        for(int j = begin; j < end; ++j)
            temp.matrix.push_back(matrix.at((i * columns) + j));

    return temp;
}

template <typename T>
Vec<T> Mat<T>::toVector()
{
    Vec<T> temp;
    if(columns == 1)
    {
        temp.rows = rows;
        temp.columns = columns;
        temp.vector = matrix;
    }
    //else exit(-1);
    return temp;
}

template <typename T>
void Mat<T>::print()
{
    for(int i = 0; i < rows; ++i)
    {
        std::cout << "[ ";
        for(int j = 0; j < columns; ++j)
            std::cout << matrix.at((i * columns) + j) << ", ";
        std::cout << "]" << std::endl;
    }
}

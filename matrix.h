#include <memory>
class Matrix{
    private:
        int row;
        int col;
        std::shared_ptr<double[]> data;
    public:
        Matrix(int r, int c);
        Matrix(const Matrix & m);
        ~Matrix();
        Matrix & operator=(const Matrix & m);
        Matrix operator+(const Matrix & m) const;
        double& operator()(int i, int j) const;
        friend std::ostream & operator<<(std::ostream & os, const Matrix & m);
};
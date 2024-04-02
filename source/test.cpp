
#include "Kernel/bubble.H"
#include <iostream>


int main(int argc, char **argv)
{
    vector<double> x={0.0, 1.0, 2.0, 3.0};
    barycentric_rational<double> bounce_x(x.data(), x.data(), x.size());
    barycentric_rational<double> bounce_p(x.data(), x.data(), x.size());
    double wid_s,  wid_l, ini_phi;
    bubble_struc(0.1, bounce_x, bounce_p, wid_s,  wid_l, ini_phi);
    for(double t=0.001;t<=10;t+=0.001){
        cout << t << '\t' << bounce_x(t) << endl; 
    }
    
}
// g++ test.cpp -g -o test
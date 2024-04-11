#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <mpi.h>

#include <AMReX_Box.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_Geometry.H>


#include <boost/numeric/odeint.hpp>
#include <boost/math/interpolators/barycentric_rational.hpp>

#include <MainCore.H>
#include <Kernels.H>

using namespace std;
using namespace amrex;
using namespace boost::numeric::odeint;
using namespace boost::math::interpolators;

// type definition

typedef std::vector<double> state_type;
typedef runge_kutta_dopri5< state_type > dopri5_type;
typedef controlled_runge_kutta< dopri5_type > controlled_dopri5_type;
typedef dense_output_runge_kutta< controlled_dopri5_type > dense_output_dopri5_type;

double kappa_global;
//boost::math::interpolators::barycentric_rational<double> bounce_x(list_0.data(), list_0.data(), list_0.size());


// integrate_observer
struct push_back_state_and_time
{
    std::vector< double >& m_states_first;
    std::vector< double >& m_states_second;
    std::vector< double >& m_times;

    push_back_state_and_time( std::vector< double > &states_first ,std::vector< double > &states_second , std::vector< double > &times )
    : m_states_first( states_first ) ,m_states_second( states_second ) , m_times( times ) { }

    void operator()( const state_type &x , double t )
    {
        m_states_first.push_back( x[0] );
        m_states_second.push_back( x[1] );
        m_times.push_back( t );
    }
};


struct push_back_state_and_time_column
{
    std::vector< double >& m_states;
    std::vector< double >& m_times;

    push_back_state_and_time_column( std::vector< double > &states , std::vector< double > &times )
    : m_states( states ) , m_times( times ) { }

    void operator()( const state_type &x , double t )
    {
        m_states.push_back( x[0] );
        m_times.push_back( t );
    }
};


// bounce equation outside the bubble
void bounce_outside(const state_type &x, state_type &dxdt, double t)
{
    dxdt[0] = x[1];
    dxdt[1] = - 3 * x[1] / t + (x[0] - pow(x[0], 3.0) + kappa_global * pow(x[0], 5.0));    
}


// calculate the initial field value of the bounce solution
double MainCore::bubble_initial_condition()
{
    dense_output_dopri5_type dopri5 = make_dense_output( 1E-14 , 1E-14 , dopri5_type() );
    double true_vac;
    double left_point, right_point, middle_point;
    double min_phi;
    vector<double> x_vec;
    vector<double> times;

    kappa_global = kappa;
    true_vac = sqrt((1 + sqrt(1 - 4 * kappa)) / 2 / kappa);
    left_point = sqrt((1 - sqrt(1 - 4 * kappa)) / 2 / kappa);
    right_point = (true_vac - left_point) * 0.9 + left_point;
    for(int i=0; i<100; i++){
        state_type x = {right_point , 0.00};
        x_vec = {};
        times = {};
        size_t steps = integrate_adaptive(dopri5, bounce_outside, x , 0.001 , 25.0 , 0.0001, push_back_state_and_time_column( x_vec , times ));
        min_phi = *min_element(x_vec.begin(), x_vec.end());
        if(min_phi>0){
            right_point = (true_vac + right_point) / 2;
        }
        else{break;}
    }
    for(int i=0; i<100; i++){
        middle_point = (left_point + right_point) / 2;
        state_type x = {middle_point , 0.00};
        x_vec = {};
        times = {};
        size_t steps = integrate_adaptive(dopri5, bounce_outside, x , 0.001 , 25.0 , 0.0001, push_back_state_and_time_column( x_vec , times ));
        min_phi = *min_element(x_vec.begin(), x_vec.end());
        if(min_phi >= 0){
            left_point = middle_point;
        }
        else{
            right_point = middle_point;
        }
        if(left_point - right_point <= 1e-10 && left_point - right_point >= -1e-10){break;}
    }
    return middle_point;
}


// search for the distance for typical phi value
double cross_point(barycentric_rational<double>& bounce_x, double phi, double start, double end)
{
    double middle;
    while(end - start > 1e-4){
        middle = (start + end) / 2;
        double phi_v = bounce_x(middle);
        if(phi_v >= phi){
            start = middle;
        }
        else{
            end = middle;
        }
    }
    return middle;
}


// get the bounce configuration and the bubble wall width
void MainCore::bubble_struc()
{
    dense_output_dopri5_type dopri5 = make_dense_output( 1E-14 , 1E-14 , dopri5_type() );
    ini_phi = bubble_initial_condition();
    state_type x = {ini_phi , 0.00};
    vector<double> x_phi;
    vector<double> p_phi;
    vector<double> times;
    size_t steps = integrate_adaptive(dopri5, bounce_outside, x , 0.001 , 25.0 , 0.0001, push_back_state_and_time( x_phi, p_phi , times ));
    bounce_x = barycentric_rational<double>(times.data(), x_phi.data(), times.size());
    bounce_p = barycentric_rational<double>(times.data(), p_phi.data(), times.size());
    double r_left = cross_point(bounce_x, ini_phi * 3 / 4, 0.001, 25.0);
    double r_right = cross_point(bounce_x, ini_phi / 4, 0.001, 25.0);
    double delta = r_right - r_left;
    wid_s = 1.5 * delta + r_right;
    wid_l = 3 * delta + r_right;
}


// minimal distance between the bubble center and the box region
Real min_distance(std::vector<double> const& list, std::vector<double>& posi_lo, 
                                                std::vector<double>& posi_hi, double L)
{
    Real distance, radius_i;
    distance = 0;
    for(int i=0;i<3;++i){
        if(posi_lo[i] <= list[i] && list[i] <= posi_hi[i]){
            radius_i = 0;
        }
        else{
            double lo = std::min(L - std::abs(posi_lo[i] - list[i]), std::abs(posi_lo[i] - list[i]));
            double hi = std::min(L - std::abs(posi_hi[i] - list[i]), std::abs(posi_hi[i] - list[i]));
            radius_i = std::min(lo, hi);
        }
        distance += radius_i * radius_i;
    }
    return distance;
}




// initialize the bubble, use trival value
void MainCore::initbubble(amrex::Box const& bx, amrex::Array4<amrex::Real> const& phi,
         amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& prob_lo,
         amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dx)
{
    using namespace amrex;

    const auto lo = lbound(bx);
    const auto hi = ubound(bx);
    dt_b = dt_0;

    const auto box_0 = Geom(0).Domain();
    const auto dx_0 = Geom(0).CellSize();

    double L = box_0.length(0) * dx_0[0];   //length of box

    int list_length = position_list_gather.size();
    double refer_dis = pow(wid_l / a_new[0], 2); //the comoving boundary of bubble
    double refer_center = pow(dt_0 / 2 + dt_b, 2); // the comoving center of bubble
    double refer_s = refer_dis + dt_b * dt_b; // the comoving boundary s value of bubble
    double a4 = pow(a_new[0], 4) * dt_0; // a ** 4 * t0

    std::vector<double> posi_lo = {prob_lo[0] + dx[0] * lo.x, 
                                    prob_lo[1] + dx[1] * lo.y, 
                                    prob_lo[2] + dx[2] * lo.z};
    std::vector<double> posi_hi = {prob_lo[0] + dx[0] * hi.x, 
                                prob_lo[1] + dx[1] * hi.y, 
                                prob_lo[2] + dx[2] * hi.z};

    
    for(int num=0;num<list_length;++num){
        double distance = min_distance(position_list_gather[num], posi_lo, posi_hi, L);
        if(distance <= refer_dis){
            double sin_ = sin(position_list_gather[num][3]);
            double cos_ = cos(position_list_gather[num][3]);
            for(int k = lo.z; k <= hi.z; ++k) {
                double z_posi = prob_lo[2] + (0.5+k) * dx[2];
                double z_dis = std::min(L - std::abs(z_posi - position_list_gather[num][2]), 
                                    std::abs(z_posi - position_list_gather[num][2]));
                for(int j = lo.y; j <= hi.y; ++j) {
                    double y_posi = prob_lo[1] + (0.5+j) * dx[1];
                    double y_dis =std::min(L - std::abs(y_posi - position_list_gather[num][1]), 
                                    std::abs(y_posi - position_list_gather[num][1]));
                    for(int i = lo.x; i <= hi.x; ++i) {
                        double x_posi = prob_lo[0] + (0.5+i) * dx[0];
                        double x_dis = std::min(L - std::abs(x_posi - position_list_gather[num][0]), 
                                        std::abs(x_posi - position_list_gather[num][0]));
                        double r_dis = x_dis * x_dis + y_dis * y_dis + z_dis * z_dis;
                        if(r_dis < refer_center){
                            phi(i, j, k, 0) = ini_phi * sin_;
                            phi(i, j, k, 1) = ini_phi * cos_;
                        }
                        else if(r_dis < refer_s){
                            double r_s = a_new[0] * sqrt(r_dis - dt_0 * dt_0);
                            phi(i, j, k, 0) = bounce_x(r_s) * sin_;
                            phi(i, j, k, 1) = bounce_x(r_s) * cos_;
                            phi(i, j, k, 2) = -bounce_p(r_s) * sin_ / r_s * a4 ;
                            phi(i, j, k, 3) = -bounce_p(r_s) * cos_ / r_s * a4 ;
                        }
                        else{}
                    }
                }
            }
        }
    }

}



// decide whether a bubble can be generated
void MainCore::judgebubble(amrex::Box const& bx, amrex::Array4<amrex::Real> const& phi,
         amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& prob_lo,
         amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dx, 
         std::vector<std::vector<double>> const& position_list, double L)
{
    using namespace amrex;

    const auto lo = lbound(bx);
    const auto hi = ubound(bx);
    double true_vac = sqrt((1 + sqrt(1 - 4 * kappa)) / 2 / kappa);
    double refer_dis = pow(wid_s / a_new[0], 2);
    double refer_phi = pow(true_vac / 10, 2);

    int list_length = position_list.size();

    std::vector<double> posi_lo = {prob_lo[0] + dx[0] * lo.x, 
                                    prob_lo[1] + dx[1] * lo.y, 
                                    prob_lo[2] + dx[2] * lo.z};
    std::vector<double> posi_hi = {prob_lo[0] + dx[0] * hi.x, 
                                prob_lo[1] + dx[1] * hi.y, 
                                prob_lo[2] + dx[2] * hi.z};
    for(int num=0;num<list_length;++num){
        double distance = min_distance(position_list[num], posi_lo, posi_hi, L);
        if(signal_list[num]==1 && distance <= refer_dis){
            for(int k = lo.z; k <= hi.z && signal_list[num]==1; ++k) {
                double z_posi = prob_lo[2] + (0.5+k) * dx[2];
                double z_dis = std::min(L - std::abs(z_posi - position_list[num][2]), 
                                    std::abs(z_posi - position_list[num][2]));
                for(int j = lo.y; j <= hi.y && signal_list[num]==1; ++j) {
                    double y_posi = prob_lo[1] + (0.5+j) * dx[1];
                    double y_dis =std::min(L - std::abs(y_posi - position_list[num][1]), 
                                    std::abs(y_posi - position_list[num][1]));
                    for(int i = lo.x; i <= hi.x && signal_list[num]==1; ++i) {
                        double phi_abs = pow(phi(i, j, k, 0), 2) + pow(phi(i, j, k, 1), 2);
                        if(phi_abs > refer_phi){
                            double x_posi = prob_lo[0] + (0.5+i) * dx[0];
                            double x_dis = std::min(L - std::abs(x_posi - position_list[num][0]), 
                                        std::abs(x_posi - position_list[num][0]));
                            double r_dis = x_dis * x_dis + y_dis * y_dis + z_dis * z_dis;
                            if(r_dis <= refer_dis){
                                signal_list[num] = 0;
                            }
                        }
                    }
                }
            }
        }
    }
}


// initialize the bubble, use trival value
void MainCore::initbubble_0(amrex::Box const& bx, amrex::Array4<amrex::Real> const& phi,
         amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& prob_lo,
         amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dx,
         std::vector<std::vector<double>> const& position_list)
{
    using namespace amrex;

    const auto lo = lbound(bx);
    const auto hi = ubound(bx);
    dt_b = dt_0;

    const auto box_0 = Geom(0).Domain();
    const auto dx_0 = Geom(0).CellSize();

    double L = box_0.length(0) * dx_0[0];   //length of box

    int list_length = position_list.size();
    double refer_dis = pow(wid_l / a_new[0], 2); //the comoving boundary of bubble
    double refer_center = pow(dt_0 / 2 + dt_b, 2); // the comoving center of bubble
    double refer_s = refer_dis + dt_b * dt_b; // the comoving boundary s value of bubble
    double a4 = pow(a_new[0], 4) * dt_0; // a ** 4 * t0

    std::vector<double> posi_lo = {prob_lo[0] + dx[0] * lo.x, 
                                    prob_lo[1] + dx[1] * lo.y, 
                                    prob_lo[2] + dx[2] * lo.z};
    std::vector<double> posi_hi = {prob_lo[0] + dx[0] * hi.x, 
                                prob_lo[1] + dx[1] * hi.y, 
                                prob_lo[2] + dx[2] * hi.z};
    for(int num=0;num<list_length;++num){
        double distance = min_distance(position_list[num], posi_lo, posi_hi, L);
        if(distance <= refer_dis){
            double sin_ = sin(position_list[num][3]);
            double cos_ = cos(position_list[num][3]);
            for(int k = lo.z; k <= hi.z; ++k) {
                double z_posi = prob_lo[2] + (0.5+k) * dx[2];
                double z_dis = std::min(L - std::abs(z_posi - position_list[num][2]), 
                                    std::abs(z_posi - position_list[num][2]));
                for(int j = lo.y; j <= hi.y; ++j) {
                    double y_posi = prob_lo[1] + (0.5+j) * dx[1];
                    double y_dis =std::min(L - std::abs(y_posi - position_list[num][1]), 
                                    std::abs(y_posi - position_list[num][1]));
                    for(int i = lo.x; i <= hi.x; ++i) {
                        double x_posi = prob_lo[0] + (0.5+i) * dx[0];
                        double x_dis = std::min(L - std::abs(x_posi - position_list[num][0]), 
                                        std::abs(x_posi - position_list[num][0]));
                        double r_dis = x_dis * x_dis + y_dis * y_dis + z_dis * z_dis;
                        if(r_dis < refer_center){
                            phi(i, j, k, 0) = ini_phi * sin_;
                            phi(i, j, k, 1) = ini_phi * cos_;
                        }
                        else if(r_dis < refer_s){
                            double r_s = a_new[0] * sqrt(r_dis - dt_0 * dt_0);
                            phi(i, j, k, 0) += bounce_x(r_s) * sin_;
                            phi(i, j, k, 1) += bounce_x(r_s) * cos_;
                            phi(i, j, k, 2) -= bounce_p(r_s) * sin_ / r_s * a4 ;
                            phi(i, j, k, 3) -= bounce_p(r_s) * cos_ / r_s * a4 ;
                        }
                        else{}
                    }
                }
            }
        }
    }

}

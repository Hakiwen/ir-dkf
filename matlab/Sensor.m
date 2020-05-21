classdef Sensor < handle
    %UNTITLED3 Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        
        A;                          % Task's discrete dynamics
        B;                          % Task's noise control matrix
        dt;                         % Time step
        R                           % Measure covariance
        Q                           % Task covariance
        
        z = 0;                      % Most recent measurement
        H;                          % Measurement matrix
        P = 1*eye(2);               % Error covariance
        x_bar;                      % This node's estimate
        u;
        U;
        message_in;                 % List of messages coming in from neighbors
        message_out;                % Message to be sent to neighbors (u_i, U_i, xi)
        
    end
    
    methods
        function s = Sensor(id, H, dt, A, B, Q, n)
            
            s.dt = dt;
            s.A = A;
            s.B = B;
            s.Q = Q;
            s.H = H;
            s.R = 30*sqrt(id);
            s.U = 1/s.R * s.H' * s.H;
            s.x_bar = [15; -10] + s.P*randn([2,1]);
            s.message_in = cell(n,1);
            s.message_out.u = 1/s.R * s.H' * s.z;
            s.message_out.U = 1/s.R * s.H' * s.H;
            s.message_out.x_bar = s.x_bar;
        end
        
        
        function update(s)
            
            % message_in{i} = (uj, Uj, xj)
            m = size(s.message_in,2);           % number of neighbors
            
            y = s.u;
            S = s.U;
            w_sum = 0;
            for j = 1:m
               y = y + s.message_in{j}.u;
               S = S + s.message_in{j}.U;
               w_sum = w_sum + ( s.message_in{j}.x_bar - s.x_bar);
            end
            
            M = inv((inv(s.P) + S));
            x_hat = s.x_bar + M * (y - S*s.x_bar) + s.dt * M * w_sum;   % prediction
            s.P = s.A*M*s.A + s.B*s.Q*s.B';                              % covariance update
            s.x_bar = s.A*x_hat;                                            % correction
            
            s.message_out.u = s.u;
            s.message_out.U = s.U;
            s.message_out.x_bar = s.x_bar;
            
        end
        
        
        function take_measurment(s,x_true)
            w = s.R*randn(1);
            s.z = s.H*x_true + w;
            
            s.u = 1/s.R * s.H' * s.z;
        end
    end
    
end


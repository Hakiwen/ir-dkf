close all; clear all; clc

dt = 0.0015;               % Time step
max_iter = 100;
figure(1), hold on, axis(500.*[-1,1,-1,1])

%% Task Initialization
A0 = 4*[0, -1; 1, 0];       % Task's continuous time dynamics
B0 = 25*eye(2);

% Discretization
A = eye(2) + dt*A0 + (dt^2/2).*A0^2 + (dt^3/6).*A0^3;
B = dt*B0;
Q = 15*eye(2);
x = [140;0];

%% Sensors Initialization
n = 8; % number of sensor
% xs = 20 - 40*rand([2,n]); % randomly locating the rpi +- 20

[xs, H, Adj, E, th] = sensorsCircConfig(n);
%[xs, H, Adj, E] = sensorsSquareConfig(n);

sensor = cell(n,1);
for i = 1:n
    sensor{i,1} = Sensor_init(i, H{i}, dt, A, B, Q, n, xs(:,i), th(i)); 
end

%% Plot Stuff
for j = 1:size(E,1) % plot egdes
    plot( [xs(1,E(j,1)),xs(1,E(j,2))] , [xs(2,E(j,1)),xs(2,E(j,2))],'-','color',[0,0,0],'markeredgecolor','none')
end

% Plot Target
thr = 0:2*pi/50:2*pi-2*pi/50;
trgt_pts = [25.*cos(thr);25.*sin(thr)];
xTrg = patch( x(1)+trgt_pts(1,:) , x(2)+trgt_pts(2,:), [1, 1, 1]);
set(xTrg,'markeredgecolor','black','LineWidth',3);
axis square

% Sensors Individual Estimates
x_bar_Pl = cell(n,1);
for i = 1:n % plot estimates
    x_bar_Pl{i} = plot(sensor{i}.z(1),sensor{i}.z(2), 'o', 'markerfacecolor',[0,1,0],'markeredgecolor','none');
end

 % Sensors plot
sensor_plot = cell(n,1);
sens_black = 2.*[-10, 10, 10, -10; -10, -10, 10, 10];
sens_green = [-40, -20, -20, -40; -40, -40, 40, 40];
sens_cones = [0, 600, 600; 0, -2.4, 2.4];
for i = 1:n
	sensor_plot{i}.black = patch(sens_black(1,:),sens_black(2,:),[0,0,0]);
    sensor_plot{i}.green = patch(sens_green(1,:),sens_green(2,:),[0,1,0]);
    sensor_plot{i}.cones = patch(sens_cones(1,:),sens_cones(2,:),[1,0,0]);
       
    sensor_plot{i}.black.XData = sensor_plot{i}.black.XData + xs(1,i);
    sensor_plot{i}.black.YData = sensor_plot{i}.black.YData + xs(2,i);
       
    sensor_plot{i}.green.XData = sensor_plot{i}.green.XData + xs(1,i);
    sensor_plot{i}.green.YData = sensor_plot{i}.green.YData + xs(2,i);
       
    sensor_plot{i}.cones.XData = sensor_plot{i}.cones.XData + xs(1,i);
    sensor_plot{i}.cones.YData = sensor_plot{i}.cones.YData + xs(2,i);
    sensor_plot{i}.cones.FaceAlpha = 0.3;
    sensor_plot{i}.cones.EdgeColor = 'none';
       
    rotate( sensor_plot{i}.black, [0,0,1], rad2deg(th(i)) , [xs(1,i),xs(2,i),0])
    rotate( sensor_plot{i}.green, [0,0,1], rad2deg(th(i)) , [xs(1,i),xs(2,i),0])
    rotate( sensor_plot{i}.cones, [0,0,1], rad2deg(th(i)) , [xs(1,i),xs(2,i),0])
end

%% Run Simulation
for k = 1:max_iter
    
    % Update task
    w = Q*randn([2,1]);
%     x = A*x + B*w;
    
    % Update
    for i = 1:n
        sensor{i} = sensor_measure(sensor{i},x); % Record new measurments
    end
   
%     set(xTrg,'XData',x(1)+trgt_pts(1,:), 'YData',x(2)+trgt_pts(2,:))
    for i = 1:n
         set(x_bar_Pl{i}, 'xdata', sensor{i}.z(1), 'ydata', sensor{i}.z(2));
    end
    drawnow
    
    pause(0.001)
    
end

function s = Sensor_init(id, H, dt, A, B, Q, n, xs, th)
    s.xs = xs;
	s.P = eye(2);
    s.z = [0;0];
    s.th = th;      % sensor orientation in global ref
    s.u = 0;
    s.dt = dt;
    s.A = A;
    s.B = B;
    s.Q = Q;
    s.H = H;
    s.R = 20*sqrt(id);
    s.U = 1/s.R * s.H' * s.H;
    s.x_bar = [15; -10];% + s.P*randn([2,1]);
    s.message_in = cell(n,1);
    s.message_out.u = 1/s.R * s.H' * s.z;
    s.message_out.U = 1/s.R * s.H' * s.H;
    s.message_out.x_bar = s.x_bar;
end   
        
function s = sensor_measure(s, xtrue)
%     w = s.R*randn(1);               % Sample noise
    
    % convert real measure into sensor distance measurment
    x_local = xtrue(1,1) - s.xs(1,1);
    y_local = xtrue(2,1) - s.xs(2,1);
    
    % verify if target is intersecting sensor's beam
%     p1 = s.xs;
%     p2 = s.xs + 900.*[ cos(s.th); sin(s.th) ];
%     dist = abs( ( p2(2)-p1(2) )*xtrue(1) - ( p2(1)-p1(1) )*xtrue(2) + p2(1)*p1(2) - p2(2)*p1(1) ) / sqrt( ( p2(2)-p1(2) )^2 + ( p2(1)-p1(1) )^2 );
    
%     if dist <= 25                                           % if robots is within range
%         ys = sqrt(x_local^2 + y_local^2);                 % actual range measure on sensor
%         x_global = s.xs + [ ys*cos(s.th); ys*sin(s.th)];    % what sensor thinks is global position of target
%     else
%         x_global = s.x_bar;
%     end
    n = 8;
    phi = 0 : 2*pi/n : 2*pi-2*pi/n;
    ys = sqrt(x_local^2 + y_local^2); 
    x_global = s.xs + [ ys*cos(phi); ys*sin(phi)]; 
    s.z = x_global;% + w;
    %x_global = s.xs + [ x_local; y_local];
    
    
    s.u = 1/s.R * s.H' * s.z;
end

function [xs, H, Adj, E, th] = sensorsCircConfig(n)
    % xs: sensors' positions in global ref
    % th: sensors' orientations in global ref
    
    sensors_array_radius = 400;                         % sensor-to-center distance
    phi = 0 : 2*pi/n : 2*pi-2*pi/n;
    alpha_offset = 20;                                  % sensor offset angle (to avoid sensors interferying with each other)
    th = phi + pi + deg2rad(alpha_offset);
    
    xs = sensors_array_radius.*[cos(phi); sin(phi)];    % sensors positions   
            
    Adj = zeros(n,n);
    for i = 1:n
        if i == 1
            Adj([2,3,7,8],i) = 1;
        elseif i == 2
             Adj([1,3,4,8],i) = 1;
        elseif i == 6
            Adj([4,5,7,8],i) = 1;
        elseif i == 7
            Adj([5,6,8,1],i) = 1;
        else
            ngb = mod(i + [-2,-1,1,2],n);
            Adj(ngb,i) = 1;
        end
    end
    
    [e1,e2] = find(tril(Adj));
    E = [e2,e1];
    
    H = cell(n,1);
    for i = 1:n
       H{i} = eye(2); 
    end    
end
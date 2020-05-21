close all; clear all; clc

dt = 0.0015;               % Time step
max_iter = 5000;

%% Task Initialization
A0 = 2*[0, -1; 1, 0];       % Task's continuous time dynamics
B0 = 25*eye(2);

% Discretization
A = eye(2) + dt*A0 + (dt^2/2).*A0^2 + (dt^3/6).*A0^3;
B = dt*B0;
Q = 0.18*eye(2);
x = [15;-10]/2;

%% Sensors Initialization
n = 8; % number of sensor
% xs = 20 - 40*rand([2,n]); % randomly locating the rpi +- 20

% toplogy (the first agent starts at the top, then counting for the next one in CW direction)
proximity_radius = 20;
xs = proximity_radius*[0 1/sqrt(2) 1 1/sqrt(2) 0 -1/sqrt(2) -1 -1/sqrt(2);
                       1 1/sqrt(2) 0 -1/sqrt(2) -1 -1/sqrt(2) 0 1/sqrt(2)];

Adj = zeros(n,n); % adjacency matrix
E = []; % neighboring edge
for i = 1:n % for each agent
    for j = i+1:n % for each other agents after i^th agent
        if (xs(:,i)-xs(:,j))'*(xs(:,i)-xs(:,j)) <= (proximity_radius*3/2)^2 % if within delta-disk graph that includes neighbors' neighbors
            % fuck yeah, neighbor!
            Adj(i,j) = 1;
            Adj(j,i) = 1;
            E = [E; i,j];
        end
    end
end

L = diag(sum(Adj)) - Adj; % Laplacian matrix
lambdas = sort(eig(L)); % for checking disconnectivity
assert(min(lambdas(2:end)>1e-5),'Graph is disconnected, try again')

sensor = cell(n,1);
for i = 1:n
%     c = rand(1);
%     if c > 0.5
    if mod(i,2) == 0
        H = [1,0]; % even agents measure x position
    else
        H = [0,1]; % odd agents measure y position
    end
    % we should change this to each agent measure the distance from itself
    % to an object in agent's local coordinate instead of directly getting
    % a global x or y coordinate
    sensor{i,1} = Sensor_init(i, H, dt, A, B, Q, n, xs(:,i)); 
end

%% Plot Stuff
figure(1), hold on
for j = 1:size(E,1) % plot egdes
    plot( [xs(1,E(j,1)),xs(1,E(j,2))] , [xs(2,E(j,1)),xs(2,E(j,2))],'-','color',[0,0,0],'markeredgecolor','none')
end

plot(xs(1,:) , xs(2,:) , 's', 'markerfacecolor',[0,0,1],'markeredgecolor','none'); % plot agents
xPl = plot(x(1),x(2), 'o', 'markerfacecolor',[1,0,0],'markeredgecolor','none'); % plot object
axis(30.*[-1 1 -1 1])
axis square

% Sensors Individual Estimates
x_bar_Pl = cell(n,1);
for i = 1:n % plot estimates
    x_bar_Pl{i} = plot(sensor{i}.x_bar(1),sensor{i}.x_bar(2), 'o', 'markerfacecolor',[0,1,0],'markeredgecolor','none');
end

%% Run Simulation
for k = 1:max_iter
    
    % Update task
    w = Q*randn([2,1]);
    x = A*x + B*w;
    
    % Update
    for i = 1:n
        sensor{i} = sensor_measure(sensor{i},x); % Record new measurments
        
        neighbors = find(Adj(i,:));
        sensor{i}.message_in(:) = [];
        sensor{i}.message_in = cell([1,size(neighbors,2)]);
        for j = 1: size(neighbors,2)
            sensor{i}.message_in{1,j} = sensor{j}.message_out;    % Creates messages in
        end
    end
    
    for i = 1:n
        sensor{i} = filter_update(sensor{i});
    end
   
    set(xPl,'xdata',x(1),'ydata',x(2))
    for i = 1:n
         set(x_bar_Pl{i}, 'xdata', sensor{i}.x_bar(1), 'ydata', sensor{i}.x_bar(2));
    end
    drawnow
    
    pause(0.001)
    
end

function s = Sensor_init(id, H, dt, A, B, Q, n, xs)
    s.xs = xs;
	s.P = eye(2);
    s.z = 0;
    s.u = 0;
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

function s = filter_update(s)
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
        
function s = sensor_measure(s,x_true)
    w = s.R*randn(1);
    s.z = s.H*x_true + w;
    s.u = 1/s.R * s.H' * s.z;
end
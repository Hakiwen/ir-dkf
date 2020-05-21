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
x = [15;-10];

%% Sensors Initialization
n = 16; % number of sensor
xs = 20 - 40*rand([2,n]); % randomly locating the sensors
proximity_radius = 16; % delta-disk graph

Adj = zeros(n,n); % adjacency matrix
E = []; % neighboring edge
for i = 1:n % for each agent
    for j = i+1:n % for each other agents after i^th agent
        if (xs(:,i)-xs(:,j))'*(xs(:,i)-xs(:,j)) <= proximity_radius^2 % if within delta-disk graph
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
    c = rand(1);
    if c > 0.5
        H = [0,1];
    else
        H = [1,0];
    end
    sensor{i,1} = Sensor(i, H, dt, A, B, Q, n); 
end

%% Plot Stuff
figure(1), hold on
for j = 1:size(E,1)
    plot( [xs(1,E(j,1)),xs(1,E(j,2))] , [xs(2,E(j,1)),xs(2,E(j,2))],'-','color',[0,0,0],'markeredgecolor','none')
end
plot(xs(1,:) , xs(2,:) , 's', 'markerfacecolor',[0,0,1],'markeredgecolor','none');
xPl = plot(x(1),x(2), 'o', 'markerfacecolor',[1,0,0],'markeredgecolor','none');
axis(30.*[-1 1 -1 1])

% Sensors Individual Estimates
x_bar_Pl = cell(n,1);
for i = 1:n
    x_bar_Pl{i} = plot(sensor{i}.x_bar(1),sensor{i}.x_bar(2), 'o', 'markerfacecolor',[0,1,0],'markeredgecolor','none');
end


%% Run Simulation
for k = 1:max_iter
    
    % Update task
    w = Q*randn([2,1]);
    x = A*x + B*w;
    
    % Update
    for i = 1:n
        sensor{i}.take_measurment(x);                           % Record new measurments
        
        neighbors = find(Adj(i,:));
        sensor{i}.message_in(:) = [];
        sensor{i}.message_in = cell([1,size(neighbors,2)]);
        for j = 1: size(neighbors,2)
            sensor{i}.message_in{1,j} = sensor{j}.message_out;    % Creates messages in
        end
    end
    
    for i = 1:n
        sensor{i}.update();
    end
   
    set(xPl,'xdata',x(1),'ydata',x(2))
    for i = 1:n
         set(x_bar_Pl{i}, 'xdata', sensor{i}.x_bar(1), 'ydata', sensor{i}.x_bar(2));
    end
    drawnow
    
    pause(0.001)
    
end

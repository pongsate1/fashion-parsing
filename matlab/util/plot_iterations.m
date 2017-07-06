function plot_iterations(filename)
%PLOT_ITERATIONS Take a log file and plot the loss over iterations.
%
% plot_iterations('log/train-fcn.o1962');
%
  patterns = {...
    '^I\d+ ([0-9:\.]+).*Iteration (\d+), loss = ([0-9\.]+)$', ...
    '^I\d+ ([0-9:\.]+).*Train net output #(\d+): loss = ([0-9\.]+).*$' ...
    };
  fprintf('Scanning validation log...\n');
  validation = readLines(filename, @(line)extractIteration(line, patterns{1}));
  fprintf('Scanning training log...\n');
  training = readLines(filename, @(line)extractIteration(line, patterns{2}));
  plot(validation(:, 2), [validation(:, 3), training(:, 3)]);
end

function extraction = extractIteration(line, pattern)
%EXTRACTITERATION
  extraction = [];
  tokens = regexp(line, pattern, 'tokens');
  if ~isempty(tokens)
    tokens = [tokens{:}];
    extraction = [datenum(tokens{1}), ...
                  str2double(tokens{2}), ...
                  str2double(tokens{3})];
  end
end

function output = readLines(filename, func)
%READLINES Read and process a text file line by line.
  fprintf('Reading %s\n', filename);
  output = cell(1, 32);
  i = 1;
  fid = fopen(filename, 'r');
  try
    while ~feof(fid)
      if i > numel(output)
        output{2 * numel(output)} = []; % Double the size if exceeding.
      end
      value = func(fgetl(fid));
      if ~isempty(value)
        output{i} = value;
        i = i + 1;
        if mod(i, 1000) == 1
          fprintf('%g lines extracted...\n', i - 1);
        end
      end
    end
    output(i:end) = [];
    output = cat(1, output{:});
  catch e
    fclose(fid);
    rethrow(e);
  end
end
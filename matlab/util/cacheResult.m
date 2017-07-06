function result = cacheResult(filename, func, varargin)
%CACHERESULT Execute the given function and cache the result.
%
%  result = cacheResult('tmp/expensiveResult.mat', @expensiveFunction, arg1);
%
  if ischar(func)
    func = str2func(func);
  end
  if exist(filename, 'file')
    logger('Loading %s result from cache: %s', func2str(func), filename);
    load(filename, 'result');
  else
    result = func(varargin{:});
    logger('Saving %s result to cache: %s', func2str(func), filename);
    save(filename, 'result');
  end
end

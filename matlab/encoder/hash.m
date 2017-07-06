function number = hash(value)
%HASH Calculate a hash for a string or a numeric/logical array.
  if ~ischar(value)
    value = num2str(value);
  end
  number = typecast(int32(java.lang.String(value).hashCode()), 'uint32');
end
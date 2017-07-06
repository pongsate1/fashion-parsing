function writeTextToFile(text, filename)
%WRITETEXTTOFILE Write text data to a file.
%
%    writeTextToFile(input, filename)
  try
    fid = fopen(filename, 'w');
    fprintf(fid, '%s', text);
    fclose(fid);
  catch exception
    if exist(filename, 'file')
        delete(filename);
    end
    rethrow(exception);
  end
end

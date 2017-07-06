function digest = sha(value)
%SHA Calculate SHA1 digest.
  digest = org.apache.commons.codec.digest.DigestUtils.sha(value)';
  digest = typecast(digest, 'uint8');
end
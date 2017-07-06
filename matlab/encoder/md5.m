function digest = md5(value)
%MD5 Calculate MD5 digest.
  digest = org.apache.commons.codec.digest.DigestUtils.md5(value)';
  digest = typecast(digest, 'uint8');
end
import sqlite3

class Cache(object):
  def __init__(self, db_name, cache_size, evict_mod, use_ram):
    self.use_ram = use_ram
    self.cache_size = cache_size
    self.evict_mod = evict_mod

    self.oldest_id = 0
    self.newest_id = 0

    if use_ram:
      self.db = {}
    else:
      self.db_name = db_name + '.db'
      self.conn = _connect()
      cursor = self.conn.cursor()    
      cursor.execute("create table videochunks(id, vgg, c3d)")
  
  def __getitem__(self, id):
    if self.use_ram:
      chunk = self.db.get(id)
      return chunk
    else:
      cursor = self.conn.cursor()
      t = (id,)
      cursor.execute('SELECT * FROM videochunks WHERE id=?', t)
      results = cursor.fetchone()
      # TODO: later, return something other than this so it matches the above
      return results[1], results[2]

  def _connect(self):
    return sqlite3.connect(db_name + '.db')

  def size(self):
      return str(self.oldest_id - self.newest_id)

  def insert(self, chunk):
    if self.use_ram:
      self.db.update({
          chunk.id: chunk
      })
    else:
      # TODO: later, is self a valid way to insert vgg and c3d data?
      cursor = self.conn.cursor()
      cursor.execute(
        "INSERT INTO videochunks VALUES(" +
        "'" + chunk.id + "', '" +
        chunk.vgg_features + "', '" +
        chunk.c3d_features + "'"
        + ")"
      )
      self.conn.commit()

    self.newest_id += 1

    # Loosely evict from the cache every [evict_mod] chunks
    oldest_allowed_id = max(0, chunk.id - self.cache_size)
    should_evict = oldest_allowed_id % self.evict_mod == 0
    if should_evict:
      self._evict(oldest_allowed_id)
      self.oldest_id = oldest_allowed_id

  def _evict(self, oldest_allowed_id):
    if self.use_ram:
      keep_evicting = True
      current_id = oldest_allowed_id - 1

      while keep_evicting:
        if not self.db.pop(current_id, None):
          keep_evicting = False
        else:
          current_id -= 1
    else:
      cursor = self.conn.cursor()
      t = (oldest_allowed_id,)
      cursor.execute('DELETE FROM videochunks WHERE id<', t)
      self.conn.commit()
